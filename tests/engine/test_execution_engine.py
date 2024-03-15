import multiprocessing
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from conftest import (
    GLUEDataBuilder,
    init_profile_data,
    template_2stages,
    template_3stages,
)
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
    tag: str = "test-gpt2"
    microbatch_size: int = 1

    def init_configuration_engine(self, temp_dir: Path):
        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send([HostInfo("127.0.0.1", 2, 1234 + i) for i in range(9)])
        # port info
        pipe.send(1234)
        self.pipe = pipe

        ConfigurationEngine.create(
            child_pipe, self.rank // 2, self.rank % 2, self.tag, temp_dir
        )

    def get_plugin(self) -> HeterogeneousParallelPlugin:
        plugin = HeterogeneousParallelPlugin(
            tp_size=2,
            global_batch_size=12,
            microbatch_size=self.microbatch_size,
            precision="fp32",
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
        [{template_3stages: 3}, {template_3stages: 1, template_2stages: 3}],
        name_fn=lambda pipeline_templates: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_engine_prepare(self, pipeline_templates: dict[PipelineTemplate, int]):
        temp_dir = TemporaryDirectory()
        self.init_configuration_engine(Path(temp_dir.name))
        init_profile_data(
            Path(temp_dir.name)
            / self.tag
            / "profile"
            / f"mb_{self.microbatch_size}.csv"
        )

        plugin = self.get_plugin()
        engine = ExecutionEngine(plugin)

        global config
        model = GPT2ForSequenceClassification(config)

        dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        assert not dist.is_initialized()

        with (
            patch(
                "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
                return_value=(
                    pipeline_templates,
                    {
                        template: 12 // sum(pipeline_templates.values())
                        for template in pipeline_templates
                    },
                ),
            ),
            patch(
                "oobleck.planner.create_pipeline_templates",
                return_value=pipeline_templates.keys(),
            ),
            patch.object(engine, "_init_distributed", new=self.fake_init_distributed),
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
        [{template_3stages: 3}, {template_3stages: 1, template_2stages: 3}],
        name_fn=lambda pipeline_templates: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    @unittest.skip("Gloo does not support pipeline send/recv")
    def test_engine_execute(self, pipeline_templates: dict[PipelineTemplate, int]):
        temp_dir = TemporaryDirectory()
        self.init_configuration_engine(Path(temp_dir.name))
        init_profile_data(
            Path(temp_dir.name)
            / self.tag
            / "profile"
            / f"mb_{self.microbatch_size}.csv"
        )

        plugin = self.get_plugin()
        engine = ExecutionEngine(plugin)

        global config
        model = GPT2ForSequenceClassification(config)

        dataloader = GLUEDataBuilder("gpt2", plugin).dataloader()

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        with (
            patch(
                "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
                return_value=(
                    pipeline_templates,
                    {
                        template: 12 // sum(pipeline_templates.values())
                        for template in pipeline_templates
                    },
                ),
            ),
            patch(
                "oobleck.planner.create_pipeline_templates",
                return_value=pipeline_templates.keys(),
            ),
            patch.object(engine, "_init_distributed", new=self.fake_init_distributed),
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

        with (
            patch.object(
                model, "sync_dp_grads", wraps=model.sync_dp_grads
            ) as sync_mock,
            patch.object(
                plugin.schedule, "forward_step", wraps=plugin.schedule.forward_step
            ) as forward_mock,
            patch.object(
                plugin.schedule, "backward_step", wraps=plugin.schedule.backward_step
            ) as backward_mock,
        ):
            result = engine.execute(iterator, model, criterion, optimizer)

        assert sync_mock.call_count == 1
        assert (
            result["loss"] is not None
            if plugin.stage_manager.is_last_stage()
            else result["loss"] is None
        )

        num_microbatches = plugin.num_microbatches[
            plugin._pipeline_index_to_pipeline[plugin._pipeline_index]
        ]
        assert forward_mock.call_count == num_microbatches
        assert backward_mock.call_count == num_microbatches


instantiate_parametrized_tests(TestExecutionEngineClass)
