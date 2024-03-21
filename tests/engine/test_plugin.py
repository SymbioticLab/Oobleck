import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch.distributed as dist
from colossalai.interface import ModelWrapper, OptimizerWrapper
from conftest import (
    GLUEDataBuilder,
    init_profile_data,
    template_1stage,
    template_2stages,
    template_3stages,
)
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from transformers import (
    AutoConfig,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.dataloader import OobleckDataLoader
from oobleck.engine.plugin import OobleckPlugin

config = AutoConfig.from_pretrained("gpt2")
config.num_hidden_layers = 4


class TestOobleckPluginClass(MultiProcessTestCase):
    tag: str = "test-gpt2"
    microbatch_size: int = 1
    host_info = [HostInfo("127.0.0.1", 1, 1234 + i) for i in range(4)]
    current_world_size: int = 4
    tp_size: int = 1

    def init_configuration_engine(self, temp_dir: Path):
        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send(self.host_info)
        # port info
        pipe.send(1234)
        self.pipe = pipe

        ConfigurationEngine.create(child_pipe, self.rank, 0, self.tag, temp_dir)

    @property
    def world_size(self):
        return self.current_world_size

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        ConfigurationEngine._instance = None

    def fake_init_distributed(self):
        if dist.is_initialized():
            dist.destroy_process_group(dist.GroupMember.WORLD)

        self.rank = ConfigurationEngine._instance.rank

        print(f"dist init r={self.rank}, world={self.world_size}")
        return dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{self.file_name}",
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
        )

    def prepare(
        self, plugin: OobleckPlugin
    ) -> tuple[ModelWrapper, OptimizerWrapper, OobleckDataLoader, LRScheduler]:
        global config
        model = GPT2ForSequenceClassification(config)

        dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, _, dataloader, lr_scheduler = plugin.configure(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            lr_scheduler=lr_scheduler,
        )

        return model, optimizer, dataloader, lr_scheduler

    @parametrize(
        "pipelines, world_size, hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                [template_1stage, template_2stages],
                3,
                [1235],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
            [
                [template_1stage, template_2stages],
                3,
                [1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
            [
                [template_2stages, template_2stages],
                4,
                [1235],
                [template_1stage, template_2stages],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                [template_2stages, template_2stages],
                4,
                [1235, 1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda _, world_size, hosts_to_fail, *__: (
            "homogeneous" if world_size == 4 else "heterogeneous"
        )
        + f"_hosts_to_fail={hosts_to_fail}",
    )
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_reconfiguration(
        self,
        pipelines: list[PipelineTemplate],
        world_size: int,
        hosts_to_fail: list[int],
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        for _ in range(self.current_world_size - world_size):
            self.host_info.pop()
        self.current_world_size = world_size
        if self.rank >= world_size:
            return

        temp_dir = TemporaryDirectory()
        self.init_configuration_engine(Path(temp_dir.name))
        init_profile_data(
            Path(temp_dir.name)
            / self.tag
            / "profile"
            / f"mb_{self.microbatch_size}.csv"
        )

        configuration_engine = ConfigurationEngine.get_instance()

        with patch.object(
            configuration_engine,
            "init_distributed",
            new=self.fake_init_distributed,
        ):
            # Consume port info that is sent from agent process
            assert configuration_engine.receive_distributed_port() == 1234
            configuration_engine.init_distributed()

            plugin = OobleckPlugin(
                tp_size=self.tp_size,
                global_batch_size=12,
                microbatch_size=1,
                precision="fp32",
            )
            plugin.set_pipelines(
                pipelines, {pipeline: 12 // len(pipelines) for pipeline in pipelines}
            )

            model, optimizer, dataloader, lr_scheduler = self.prepare(plugin)

            hosts_remaining = []

            # Simulate agent process's behavior sending the new host info
            for host, ranks in list(configuration_engine.rank_map.items()):
                if host.port in hosts_to_fail:
                    if self.rank in ranks:
                        return
                else:
                    hosts_remaining.append(host)
            self.pipe.send(hosts_remaining)

            self.current_world_size = len(hosts_remaining)

            plugin.reconfigure(
                pipeline_templates={
                    1: template_1stage,
                    2: template_2stages,
                    3: template_3stages,
                },
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
            )

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)


instantiate_parametrized_tests(TestOobleckPluginClass)
