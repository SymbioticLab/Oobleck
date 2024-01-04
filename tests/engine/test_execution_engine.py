import multiprocessing
from unittest.mock import patch

import pytest
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from conftest import GLUEDataBuilder, heterogeneous_templates, homogeneous_templates
from oobleck_colossalai import (
    HeterogeneousParallelModule,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)
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

    def get_plugin(self) -> HeterogeneousParallelPlugin:
        plugin = HeterogeneousParallelPlugin(
            tp_size=2, global_batch_size=12, microbatch_size=1, precision="fp32"
        )
        return plugin

    @property
    def world_size(self):
        return 18

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        ConfigurationEngine._instance = None

    def fake_init_distributed(self):
        print(f"dist init r={self.rank}, world={self.world_size}")
        return dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{self.file_name}",
            backend="gloo",
            world_size=self.world_size,
            rank=self.rank,
        )

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

        dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        assert not dist.is_initialized()

        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                pipeline_templates,
                {
                    template: 12 // sum(pipeline_templates.values())
                    for template in pipeline_templates
                },
            ),
        ), patch(
            "oobleck.engine.execution_engine.PipelineTemplate.generate_pipeline_templates",
            return_value=pipeline_templates.keys(),
        ), patch.object(
            engine, "_init_distributed", new=self.fake_init_distributed
        ):
            model, optimizer, _, dataloader, lr_scheduler = engine.prepare(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                lr_scheduler=lr_scheduler,
            )

        assert dist.is_initialized()
        assert (
            dataloader.batch_sampler
        ), "HeterogeneousDataLoader.configure() is not called."

    @parametrize(
        "pipeline_templates",
        [homogeneous_templates, heterogeneous_templates],
        name_fn=lambda pipeline_templates: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_engine_execute(self, pipeline_templates: dict[PipelineTemplate, int]):
        self.init_configuration_engine()

        plugin = self.get_plugin()
        engine = ExecutionEngine(plugin)

        global config
        model = GPT2ForSequenceClassification(config)

        dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                pipeline_templates,
                {
                    template: 12 // sum(pipeline_templates.values())
                    for template in pipeline_templates
                },
            ),
        ), patch(
            "oobleck.engine.execution_engine.PipelineTemplate.generate_pipeline_templates",
            return_value=pipeline_templates.keys(),
        ), patch.object(
            engine, "_init_distributed", new=self.fake_init_distributed
        ):
            model, optimizer, criterion, dataloader, lr_scheduler = engine.prepare(
                model=model,
                criterion=lambda outputs, inputs: outputs.loss,
                dataloader=dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        assert isinstance(model, HeterogeneousParallelModule)
        assert isinstance(
            optimizer, (HybridParallelAMPOptimizer, HybridParallelNaiveOptimizer)
        )

        iterator = iter(dataloader)

        with patch.object(
            model, "sync_dp_grads", wraps=model.sync_dp_grads
        ) as sync_mock:
            result = engine.execute(iterator, model, criterion, optimizer)

        sync_mock.assert_called_once()
        assert (
            result["loss"] is not None
            if plugin.stage_manager.is_last_stage()
            else result["loss"] is None
        )


instantiate_parametrized_tests(TestExecutionEngineClass)
