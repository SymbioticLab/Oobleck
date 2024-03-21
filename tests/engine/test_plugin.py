import multiprocessing
import sys
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

tag: str = "test-gpt2"
microbatch_size: int = 1
global_batch_size: int = 12


class OobleckPluginClassBase(MultiProcessTestCase):
    current_world_size: int
    pipelines: list[PipelineTemplate]

    def generate_host_info(self, tp_size: int) -> list[HostInfo]:
        return [
            HostInfo("127.0.0.1", tp_size, 1234 + i) for i in range(self.world_size)
        ]

    def init_configuration_engine(self, tp_size: int, temp_dir: Path):
        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send(self.generate_host_info(tp_size))
        # port info
        pipe.send(1234)
        self.pipe = pipe

        ConfigurationEngine.create(
            child_pipe, self.rank // tp_size, self.rank % tp_size, tag, temp_dir
        )

    @property
    def world_size(self):
        return self.current_world_size

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        ConfigurationEngine._instance = None

    def init_distributed(self):
        if dist.is_initialized():
            dist.destroy_process_group(dist.GroupMember.WORLD)

        self.rank = ConfigurationEngine._instance.rank

        print(f"dist init r={self.rank}, world={self.world_size}")
        dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{self.file_name}",
            backend=None,  # both gloo and nccl is initialized
            world_size=self.world_size,
            rank=self.rank,
        )

        configuration_engine = ConfigurationEngine.get_instance()
        assert configuration_engine.rank == self.rank
        assert configuration_engine.world_size == self.world_size

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

    def init_test(
        self, tp_size: int
    ) -> tuple[OobleckPlugin, ModelWrapper, OptimizerWrapper, OobleckDataLoader]:
        temp_dir = TemporaryDirectory()
        self.init_configuration_engine(tp_size, Path(temp_dir.name))
        init_profile_data(
            Path(temp_dir.name) / tag / "profile" / f"mb_{microbatch_size}.csv"
        )

        configuration_engine = ConfigurationEngine.get_instance()

        # Consume port info that is sent from agent process
        assert configuration_engine.receive_distributed_port() == 1234
        self.init_distributed()

        plugin = OobleckPlugin(
            tp_size=tp_size,
            global_batch_size=global_batch_size,
            microbatch_size=1,
            precision="fp32",
        )
        plugin.set_pipelines(
            self.pipelines,
            {
                pipeline: global_batch_size // len(self.pipelines)
                for pipeline in self.pipelines
            },
        )

        model, optimizer, dataloader, lr_scheduler = self.prepare(plugin)

        return plugin, model, optimizer, dataloader

    def do_step(
        self,
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: OobleckDataLoader,
    ):
        iterator = iter(dataloader)
        plugin.execute_pipeline(
            iterator,
            model,
            lambda outputs, inputs: outputs.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )

    def do_reconfigure(
        self,
        hosts_to_fail: list[int],
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: OobleckDataLoader,
    ):
        configuration_engine = ConfigurationEngine.get_instance()

        # Simulate agent process's behavior sending the new host info
        hosts_remaining = []
        for host, ranks in list(configuration_engine.rank_map.items()):
            if host.port in hosts_to_fail:
                if self.rank in ranks:
                    sys.exit(0)
            else:
                hosts_remaining.append(host)
        self.pipe.send(hosts_remaining)

        # self.do_step(plugin, model, optimizer, dataloader)

        with patch.object(
            configuration_engine, "init_distributed", new=self.init_distributed
        ):
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


class TestOobleckReconfiguration3RanksClass(OobleckPluginClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_world_size = 3
        self.pipelines = [template_1stage, template_2stages]

    @parametrize(
        "tp_size, hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                1,
                [1235],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
            [
                1,
                [1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda tp_size,
        hosts_to_fail,
        *_: f"tp_size={tp_size},hosts_to_fail={hosts_to_fail}",
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_reconfiguration_pass(
        self,
        tp_size: int,
        hosts_to_fail: int,
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.init_test(tp_size)
        self.do_reconfigure(hosts_to_fail, plugin, model, optimizer, dataloader)

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)


class TestOobleckReconfiguration4RanksClass(OobleckPluginClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_world_size = 4
        self.pipelines = [template_2stages, template_2stages]

    @parametrize(
        "tp_size, hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                1,
                [1235],
                [template_1stage, template_2stages],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                1,
                [1237],
                [template_2stages, template_1stage],
                [
                    [[0], [0], [0], [1], [1], [1], [1], [1], [1]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                1,
                [1235, 1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda tp_size, hosts_to_fail, *_: (
            f"tp_size={tp_size},hosts_to_fail={hosts_to_fail}"
        ),
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_reconfiguration_pass(
        self,
        tp_size: int,
        hosts_to_fail: int,
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.init_test(tp_size)
        self.do_reconfigure(hosts_to_fail, plugin, model, optimizer, dataloader)

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)


instantiate_parametrized_tests(TestOobleckReconfiguration3RanksClass)
instantiate_parametrized_tests(TestOobleckReconfiguration4RanksClass)
