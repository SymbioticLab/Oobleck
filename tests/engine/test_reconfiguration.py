import multiprocessing
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.interface import ModelWrapper, OptimizerWrapper
from conftest import (
    config,
    template_1stage,
    template_2stages,
    template_3stages,
)
from data_builder import GLUEDataBuilder
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.optim import Adam
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.plugin import OobleckPlugin

tag: str = "test-gpt2"
microbatch_size: int = 1
global_batch_size: int = 12


class OobleckReconfigurationClassBase(MultiProcessTestCase):
    num_hosts: int
    tp_size: int = 1
    pipe: Connection
    reconfiguration_count: int = 0

    @property
    def world_size(self) -> int:
        return self.num_hosts * self.tp_size

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        ConfigurationEngine._instance = None

    def init_oobleck(self):
        torch.cuda.set_device(self.rank)

        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send(
            [
                HostInfo("127.0.0.1", self.tp_size, 1234 + i)
                for i in range(self.world_size)
            ]
        )
        # port info
        pipe.send(1234)
        self.pipe = pipe

        temp_dir = Path(TemporaryDirectory().name)

        ConfigurationEngine.create(
            child_pipe,
            self.rank // self.tp_size,
            self.rank % self.tp_size,
            tag,
            temp_dir,
        )

        # init_profile_data(temp_dir / tag / "profile" / f"mb_{microbatch_size}.csv")

        # Consume port info that is sent from agent process
        assert ConfigurationEngine.get_instance().receive_distributed_port() == 1234
        self.init_distributed()

    def init_distributed(self):
        if dist.is_initialized():
            dist.destroy_process_group(dist.GroupMember.WORLD)

        configuration_engine = ConfigurationEngine.get_instance()
        self.rank = configuration_engine.rank
        self.num_hosts = configuration_engine.world_size // self.tp_size

        print(f"dist init r={self.rank}, world={self.world_size}")

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}{self.reconfiguration_count}",
                backend="nccl",
                world_size=self.world_size,
                rank=self.rank,
            )
            self.reconfiguration_count += 1

        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        assert dist.is_initialized()

    def prepare(
        self, pipelines: list[PipelineTemplate]
    ) -> tuple[OobleckPlugin, ModelWrapper, OptimizerWrapper, DataLoader]:
        self.init_oobleck()

        templates = [template_1stage, template_2stages]

        with patch.object(
            OobleckPlugin,
            "_instantiate_pipelines",
            return_value=(
                pipelines,
                {
                    template: global_batch_size // len(templates)
                    for template in templates
                },
            ),
        ):
            plugin = OobleckPlugin(
                pipeline_templates=templates,
                tp_size=self.tp_size,
                global_batch_size=global_batch_size,
                microbatch_size=microbatch_size,
                precision="bf16",
            )

        dataloader = GLUEDataBuilder("gpt2", plugin).train_dataloader()
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, _, dataloader, lr_scheduler = plugin.configure(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            lr_scheduler=lr_scheduler,
        )

        return plugin, model, optimizer, dataloader

    def do_step(
        self,
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ):
        plugin.execute_pipeline(
            iter(dataloader),
            model,
            lambda outputs, inputs: outputs.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )

        optimizer.step()
        optimizer.zero_grad()

    def do_reconfigure(
        self,
        hosts_to_fail: list[int],
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ):
        configuration_engine = ConfigurationEngine.get_instance()

        # Simulate agent process's behavior sending the new host info
        hosts_remaining = []
        for host, ranks in configuration_engine.rank_map.items():
            if host.port in hosts_to_fail:
                if self.rank in ranks:
                    print(f"Rank {self.rank} failed")
                    sys.exit(0)
            else:
                hosts_remaining.append(host)
        self.pipe.send(hosts_remaining)
        self.num_hosts -= len(hosts_to_fail)

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


class TestOobleckReconfiguration3RanksClass(OobleckReconfigurationClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 3

    @parametrize(
        "hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                [1235],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
            [
                [1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda hosts_to_fail, *_: f"hosts_to_fail={hosts_to_fail}",
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_reconfiguration_pass(
        self,
        hosts_to_fail: int,
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_1stage, template_2stages]
        )
        self.do_step(plugin, model, optimizer, dataloader)
        self.do_reconfigure(hosts_to_fail, plugin, model, optimizer, dataloader)

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)

        self.do_step(plugin, model, optimizer, dataloader)


class TestOobleckReconfiguration4RanksClass(OobleckReconfigurationClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 4

    @parametrize(
        "hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                [1235],
                [template_1stage, template_2stages],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                [1237],
                [template_2stages, template_1stage],
                [
                    [[0], [0], [0], [1], [1], [1], [1], [1], [1]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                [1235, 1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda hosts_to_fail, *_: (f"hosts_to_fail={hosts_to_fail}"),
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_reconfiguration_pass(
        self,
        hosts_to_fail: int,
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_2stages, template_2stages]
        )
        self.do_step(plugin, model, optimizer, dataloader)

        self.do_reconfigure(hosts_to_fail, plugin, model, optimizer, dataloader)

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)


instantiate_parametrized_tests(TestOobleckReconfiguration3RanksClass)
instantiate_parametrized_tests(TestOobleckReconfiguration4RanksClass)
