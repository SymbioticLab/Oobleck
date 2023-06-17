from __future__ import annotations

import json
import os
import random
import shutil
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "microsoft/resnet-50": Model("microsoft/resnet-50", "Maysee/tiny-iamgenet"),
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


@pytest.fixture(scope="session", params=list(models_to_test.keys()))
def models_to_test(request: pytest.FixtureRequest) -> str:
    return request.param


class OobleckClassFactory:
    # This should be used via getfixturevalue in multiprocessing tests.

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
        self._profile: Optional[Tuple[List, List, List]] = None

    def get_dataset(self) -> OobleckDataset:
        if not self._dataset:
            self._dataset = OobleckDataset(
                self._model_data.model_name, self._model_data.dataset_path
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

    def get_dummy_profile(self) -> Tuple[List, List, List]:
        self.get_model()

        if not self._profile:
            num_layers = len(self._model.model)
            layers = []
            allreduce_across_nodes = []
            allreduce_in_node = []
            for _ in range(num_layers):
                layers.append(
                    {
                        "forward": random.random(),
                        "backward": random.random() * 3,
                        "mem_required": [1024, 1024],
                    }
                )

                # TODO: get argument to set number of nodes
                ar_across_nodes = {}
                for i in range(64):  # up to 64 nodes
                    ar_across_nodes[i + 1] = random.random() * 4

                allreduce_across_nodes.append(ar_across_nodes)
                allreduce_in_node.append(
                    {1: random.random(), 2: random.random(), 4: random.random()}
                )

                self._profile = (layers, allreduce_across_nodes, allreduce_in_node)

        return self._profile


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
