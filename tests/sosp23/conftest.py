import logging
import math
import random
from multiprocessing import connection
from typing import Any

import pytest
from pytest_mock import MockerFixture
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    PipelineTemplate,
    StageExecutionResult,
)
from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader, OobleckSampler
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    # Ignore all other errors: https://stackoverflow.com/a/58539831
    for _ in logging.root.manager.loggerDict:
        logging.getLogger(_).disabled = True

    logger = logging.getLogger("oobleck-sosp23")
    logger.disabled = False
    logger.setLevel(logging.INFO)
    return logger


eval_model_args: dict[str, dict[str, int]] = {
    "fake_model1": {
        "num_hidden_layers": 24,
        "n_positions": 1024,
    },
    "fake_model2": {
        "num_hidden_layers": 32,
        "n_positions": 1024,
    },
}


@pytest.fixture(scope="session", params=list(eval_model_args.keys()))
def eval_model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def eval_dataset() -> OobleckDataset:
    return OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1", 1024)


@pytest.fixture(scope="session")
def eval_model(eval_model_name: str, eval_dataset: OobleckDataset) -> OobleckModel:
    return OobleckModel(
        model_name="gpt2",  # we use HuggingFace model, thus this model should be registered in HF hub.
        sample_inputs=eval_dataset.sample,
        model_tag="test",
        config_args=eval_model_args[eval_model_name],
    )


@pytest.fixture(scope="session")
def eval_dummy_profile(eval_model_name: str) -> LayerExecutionResults:
    results: list[LayerExecutionResult] = []

    if eval_model_name == "fake_model1":
        # 24 layers with equal execution time
        for index in range(24):
            results.append(
                LayerExecutionResult(
                    layer_index=index,
                    forward=0.05,
                    backward=0.1,
                    allreduce_in_node={i + 1: 0.2 for i in range(4)},
                    allreduce_across_nodes={i + 1: 0.5 for i in range(16)},
                    mem_required=(1024, 1024),
                ),
            )
    else:
        # 32 layers with constantly increasing execution time
        for index in range(32):
            results.append(
                LayerExecutionResult(
                    layer_index=index,
                    forward=0.03 + random.random() * 0.02,
                    backward=0.08 + random.random() * 0.02,
                    allreduce_in_node={i + 1: 0.2 for i in range(4)},
                    allreduce_across_nodes={i + 1: 0.5 for i in range(16)},
                    mem_required=(1024, 1024),
                )
            )
    return LayerExecutionResults(results)


@pytest.fixture(scope="session")
def eval_pipeline_templates(
    eval_dummy_profile: LayerExecutionResults,
) -> list[PipelineTemplate]:
    def slice_layers(lst: list[Any], num_chunks: int) -> list[tuple[int, int]]:
        if num_chunks > len(lst):
            raise ValueError(
                f"Cannot slice {len(list)} layers into {num_chunks} chunks."
            )

        length_chunk = math.ceil(len(lst) / num_chunks)
        slicing_points: list[tuple[int, int]] = []
        for i in range(0, len(lst), length_chunk):
            end = i + length_chunk if i + length_chunk < len(lst) else len(lst)
            slicing_points.append((i, end))
        return slicing_points

    results: list[PipelineTemplate] = []
    for num_stages in range(2, 6):
        num_gpus_per_node = 1
        num_nodes = num_stages

        layer_indices = slice_layers(eval_dummy_profile.get(), num_stages)

        num_gpus_per_stage = (num_nodes * num_gpus_per_node) // num_stages
        stages = [
            StageExecutionResult(eval_dummy_profile, indices, num_gpus_per_stage)
            for indices in layer_indices
        ]

        results.append(
            PipelineTemplate(
                stages,
                0.1,
                eval_dummy_profile.size,
                num_nodes,
                num_gpus_per_node,
            )
        )

    return results


@pytest.fixture(scope="module")
def sample_args() -> OobleckArguments:
    return OobleckArguments(
        model_name="fakemodel",
        model_tag="test",
        dataset_path="fake_dataset",
        dataset_name="fake_dataset_v1",
        fault_threshold=1,
        model_args={},
        microbatch_size=1,
        global_microbatch_size=24,
    )


class FakePipeline:
    def __init__(
        self,
        pipeline_id: int,
        pipeline_template: PipelineTemplate,
        ranks: list[int],
        dataloader: OobleckDataLoader,
        *args,
        **kwargs,
    ):
        self._pipeline_id = pipeline_id
        self._template = pipeline_template
        self._ranks = ranks
        self._dataloader = dataloader
        self._global_step = 0
        self.my_pipeline = bool(0 in self._ranks)

        # copy from pipeline __init__
        # layer_index -> ranks
        self.rank_grid: dict[int, list[int]] = pipeline_template.get_rank_grid(
            self._ranks
        )

    def initialize_distributed_fsdp(self, *args):
        pass

    def initialize_distributed_pipeline(self, *args):
        pass

    def initialize_execution(self, *args):
        pass


class EvalFakeEngine:
    def __init__(
        self,
        dataset: OobleckDataset,
        model: OobleckModel,
        sample_args: OobleckArguments,
        dummy_profile: LayerExecutionResults,
        pipeline_templates: list[PipelineTemplate],
    ):
        self._agent_pipe: connection.Connection = None
        self._args = sample_args
        self._hf_training_args = TrainingArguments(output_dir="/tmp/outut")
        self._num_nodes = sum(range(2, 6))
        self._num_gpus_per_node = 1
        self._dataset = dataset
        self._model = model
        self._profile_results = dummy_profile
        self._pipeline_templates = pipeline_templates

    def init_pipelines(self) -> list[OobleckPipeline]:
        num_gpus_used: int = 0
        pipeline_id: int = 0
        pipelines: list[OobleckPipeline] = []

        dataloader = OobleckDataLoader(
            self._hf_training_args, self._dataset, LoaderType.Training, 0, [0], 0, 0
        )

        # Have one pipelines for each pipeline template
        for template in self._pipeline_templates:
            pipelines.append(
                FakePipeline(
                    pipeline_id,
                    template,
                    list(
                        range(
                            num_gpus_used,
                            num_gpus_used + len(template.get_stages()),
                        )
                    ),
                    dataloader,
                )
            )

            pipeline_id += 1
            num_gpus_used += len(template.get_stages())
        self._pipeline = pipelines[0]
        return pipelines


@pytest.fixture
def fake_engine(
    mocker: MockerFixture,
    eval_dataset: OobleckDataset,
    eval_model: OobleckModel,
    sample_args: OobleckArguments,
    eval_dummy_profile: LayerExecutionResults,
    eval_pipeline_templates: list[PipelineTemplate],
) -> EvalFakeEngine:
    mocker.patch(
        "oobleck.planning.instantiator.OobleckPipeline",
        new=FakePipeline,
    )
    mocker.patch(
        "deepspeed.comm.new_group",
        return_value=None,
    )
    mocker.patch(
        "deepspeed.comm.get_rank",
        return_value=0,
    )
    mocker.patch(
        "torch.distributed.init_process_group",
        return_value=None,
    )
    mocker.patch(
        "oobleck.execution.engine.ReconfigurationEngine._copy_model_states",
        return_value=None,
    )
    yield EvalFakeEngine(
        eval_dataset,
        eval_model,
        sample_args,
        eval_dummy_profile,
        eval_pipeline_templates,
    )
