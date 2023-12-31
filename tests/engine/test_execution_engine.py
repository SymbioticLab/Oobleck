import multiprocessing
from unittest.mock import patch

import pytest
import torch.distributed as dist
from conftest import GLUEDataBuilder, heterogeneous_templates, homogeneous_templates
from oobleck_colossalai import HeterogeneousParallelPlugin, PipelineTemplate
from torch.optim import Adam
from torch.testing._internal.common_distributed import MultiProcessTestCase
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
from oobleck.engine.execution_engine import ExecutionEngine

config = AutoConfig.from_pretrained("gpt2")
config.num_hidden_layers = 4


class TestExecutionEngineClass(MultiProcessTestCase):
    def init_configuration_engine(self):
        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send([HostInfo("127.0.0.1", 2, 1234 + i) for i in range(9)])
        # port info
        pipe.send(1234)
        self.pipe = pipe

        ConfigurationEngine.create(child_pipe, self.rank // 2, self.rank % 2)
        # mock1 = patch.object(
        #     ConfigurationEngine._instance, "send_distributed_port", return_value=None
        # )
        # mock2 = patch.object(
        #     ConfigurationEngine._instance, "receive_distributed_port", return_value=1234
        # )

    def get_plugin(self) -> HeterogeneousParallelPlugin:
        plugin = HeterogeneousParallelPlugin(tp_size=2, microbatch_size=1)
        return plugin

    @property
    def world_size(self):
        return 18

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @parametrize(
        "pipeline_templates",
        [homogeneous_templates, heterogeneous_templates],
        name_fn=lambda pipeline_templates: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_engine_prepare(self, pipeline_templates: dict[PipelineTemplate, int]):
        self.init_configuration_engine()

        plugin = self.get_plugin()
        engine = ExecutionEngine(plugin)

        global config
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        assert not dist.is_initialized()

        def fake_init_distributed():
            print(f"dist init r={self.rank}, world={self.world_size}")
            return dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend="gloo",
                world_size=self.world_size,
                rank=self.rank,
            )

        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                pipeline_templates,
                {template: 1 for template in pipeline_templates},
            ),
        ), patch(
            "oobleck.engine.execution_engine.PipelineTemplate.generate_pipeline_templates",
            return_value=pipeline_templates.keys(),
        ), patch.object(
            engine, "_init_distributed", new=fake_init_distributed
        ):
            model, optimizer, _, lr_scheduler, _ = engine.prepare(
                model=model,
                criterion=lambda outputs, inputs: outputs.loss,
                optimizer=optimizer,
                scheduler=lr_scheduler,
            )

        # dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        assert dist.is_initialized()


instantiate_parametrized_tests(TestExecutionEngineClass)
