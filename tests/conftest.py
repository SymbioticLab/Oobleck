from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    PipelineTemplate,
    StageExecutionResult,
)
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@dataclass
class Model:
    model_name: str
    dataset_path: str
    dataset_name: Optional[str] = None


models_to_test: Dict[str, Model] = {
    "gpt2": Model("gpt2", "wikitext", "wikitext-2-raw-v1"),
    "microsoft/resnet-50": Model("microsoft/resnet-50", "Maysee/tiny-imagenet"),
}

# Add model arguments here, if it is needed.
model_args: Dict[str, Optional[Dict[str, int]]] = {
    "gpt2": {
        "num_hidden_layers": 32,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    },
}


@pytest.fixture(scope="class", params=list(models_to_test.keys()))
def model_name_fixture(request: pytest.FixtureRequest) -> str:
    return request.param


class OobleckStaticClassFactory:
    """
    Oobleck Class Factory that create classes for testing.
    "Static" here means that it is not relevant to Oobleck dynamic reconfiguration
    and fixed once a class object is created.
    """

    def __init__(self, model_name: str, test_directory: str):
        self._model_data: Model = models_to_test[model_name]
        self._training_args = TrainingArguments(
            output_dir=test_directory,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
        )

        self._dataset: Optional[OobleckDataset] = None
        self._model: Optional[OobleckModel] = None
        self._dataloader: Optional[OobleckDataLoader] = None
        self._profile: Optional[LayerExecutionResults] = None
        self._pipeline_template: Optional[PipelineTemplate] = None

    def get_dataset(self) -> OobleckDataset:
        if not self._dataset:
            self._dataset = OobleckDataset(
                self._model_data.model_name,
                self._model_data.dataset_path,
                self._model_data.dataset_name,
            )
        return self._dataset

    def get_model(self) -> OobleckModel:
        self.get_dataset()

        if not self._model:
            self._model = OobleckModel(
                self._model_data.model_name,
                self._dataset.sample,
                self._training_args,
                "test",
                model_args.get(self._model_data.model_name, None),
            )

        return self._model

    # TODO: move it to dynamic class factory
    # Dataloader has its state and is subject to change.
    def get_dataloader(self) -> OobleckDataLoader:
        self.get_dataset()
        self.get_model()

        if not self._dataloader:
            self._dataloader = OobleckDataLoader(
                self._dataset,
                self._training_args,
                LoaderType.Training,
                self._training_args.gradient_accumulation_steps,
                0,
                0,
            )

        return self._dataloader

    def get_dummy_profile(self) -> LayerExecutionResults:
        self.get_model()

        if not self._profile:
            num_layers = len(self._model.model)

            results: List[LayerExecutionResult] = []
            for index in range(num_layers):
                results.append(
                    LayerExecutionResult(
                        layer_index=index,
                        forward=random.random(),
                        backward=random.random() * 3,
                        allreduce_in_node={i + 1: random.random() for i in range(8)},
                        allreduce_across_nodes={
                            i + 1: random.random() * 4 for i in range(64)
                        },
                        mem_required=[1024, 1024],
                    )
                )

            self._profile = LayerExecutionResults(results)

        return self._profile

    def get_dummpy_pipeline_template(self, num_gpus: int) -> PipelineTemplate:
        def slice_layers(lst: List[Any], num_chunks: int) -> List[Tuple[int, int]]:
            if num_chunks > len(lst):
                raise ValueError(
                    f"Cannot slice {len(list)} layers into {num_chunks} chunks."
                )

            slice_points = sorted(random.sample(range(1, len(lst)), num_chunks - 1))
            slice_points = [0] + slice_points + [None]
            return [(slice_points[i], slice_points[i + 1]) for i in range(num_chunks)]

        if not self._pipeline_template:
            # TODO: take user argument for it
            num_gpus_per_stage = 1

            layer_results = self.get_dummy_profile()
            layer_indices = slice_layers(layer_results.get(), num_gpus)

            stages = [
                StageExecutionResult(
                    layer_results, layer_indices[i], num_gpus_per_stage
                )
                for i in range(layer_indices)
            ]

            self._pipeline_template = PipelineTemplate(
                stages, 0.1, layer_results.size(), num_gpus, num_gpus_per_stage
            )

        return self._pipeline_template


class OobleckSingleProcessTestCase:
    """
    A base class for Oobleck test cases that run in a single process.
    Test cases for functionalities of static classes will inherit this class.
    """

    factory: OobleckStaticClassFactory

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        model_name_fixture: str,
        tmpdir_factory: pytest.TempdirFactory,
        request: pytest.FixtureRequest,
    ):
        tmpdir = tmpdir_factory.mktemp("oobleck")
        request.cls.factory = OobleckStaticClassFactory(
            model_name_fixture, tmpdir.dirpath()
        )


class OobleckMultiProcessTestCase:
    """
    A base class for Oobleck test cases that run in multiple processes in parallel.
    Test cases for functionalities of dynamic classes will inherit this class.
    """

    pass


@pytest.fixture(scope="function")
def no_distributed():
    original_env = dict(os.environ)
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    os.environ.pop("LOCAL_RANK", None)
    next(set_number_of_gpus(1))
    yield
    os.environ.clear()
    os.environ.update(original_env)


def set_number_of_gpus(num_gpus: int = 1):
    # Hack to make torch.cuda.device_count() return # GPUs specified in env
    func = torch.cuda.device_count
    torch.cuda.device_count = lambda: num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(0, num_gpus)])
    yield
    torch.cuda.device_count = func
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


@pytest.fixture(scope="function")
def init_distributed():
    original_env = dict(os.environ)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    set_number_of_gpus(1)

    def _distributed(init_required: bool):
        if init_required:
            if dist.is_initialized():
                return

            dist.init_distributed(
                dist_backend="nccl", dist_init_required=True, rank=0, world_size=1
            )
            assert dist.is_initialized()
        else:
            assert not dist.is_initialized()

    yield _distributed

    if dist.is_initialized():
        dist.destroy_process_group()
        dist.cdb = None
    assert not dist.is_initialized()

    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def dummy_pipeline_template(dummy_layer_execution_results: LayerExecutionResults):
    def get_layer_split_indices(
        layers: List[LayerExecutionResult], num: int
    ) -> List[List[LayerExecutionResult]]:
        return [round(len(layers) * i / num) for i in range(1, num)]

    def _create_pipeline_template(num_gpus: int) -> PipelineTemplate:
        layers = dummy_layer_execution_results.get()
        layer_results = LayerExecutionResults(layers)
        indices = get_layer_split_indices(layers, num_gpus)
        stages = [
            StageExecutionResult(layer_results, indices, 1)
            for indices in zip([0] + indices, indices + [len(layers)])
        ]

        return PipelineTemplate(stages, 0.1, len(layers), num_gpus, 1)

    return _create_pipeline_template


@pytest.fixture(scope="function")
def dummy_pipeline(model_dataloaders, dummy_pipeline_template, init_distributed):
    def _create_pipelines(
        num_gpus_per_pipeline: int, ranks: List[int]
    ) -> List[OobleckPipeline]:
        assert len(ranks) % num_gpus_per_pipeline == 0, "Invalid number of ranks"
        assert len(ranks) <= torch.cuda.device_count(), "Too many ranks"

        model, train_dataloader, eval_dataloader = model_dataloaders
        init_distributed(True)
        training_args = TrainingArguments(
            output_dir="/tmp/test_output",
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
        )

        pipeline_template = dummy_pipeline_template(num_gpus=num_gpus_per_pipeline)

        rank_groups = [
            ranks[i:num_gpus_per_pipeline]
            for i in range(0, len(ranks), num_gpus_per_pipeline)
        ]

        pipelines = [
            OobleckPipeline(
                pipeline_template=pipeline_template,
                model=model,
                dataloader=train_dataloader,
                step=0,
                ranks=rank_group,
                process_group=torch.distributed.new_group(ranks=rank_group),
                training_args=training_args,
            )
            for rank_group in rank_groups
        ]

        for pipeline in pipelines:
            assert all(
                all(p.is_cuda for p in layer.parameters())
                for layer in pipeline.model_layers
            )
            assert pipeline.is_first_stage() and pipeline.is_last_stage()

            num_pipe_buffers = pipeline.train_schedule.num_pipe_buffers()
            for buffer_name in ["inputs", "labels", "outputs"]:
                assert buffer_name in pipeline.pipe_buffers
                assert len(pipeline.pipe_buffers[buffer_name]) == num_pipe_buffers
                assert all(
                    buffer is None for buffer in pipeline.pipe_buffers[buffer_name]
                )

        return pipelines

    return _create_pipelines
