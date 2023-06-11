import itertools
import json
import math
import os
import random
import shutil
import string
from pathlib import Path
from typing import List

import deepspeed.comm as dist
import pytest
from transformers import TrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
    StageExecutionResult,
)
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.module.model import OobleckModel

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@pytest.fixture(scope="session")
def wikitext_dataset():
    return OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")


@pytest.fixture(scope="session")
def imagenet_dataset():
    return OobleckDataset("microsoft/resnet-50", "Maysee/tiny-imagenet")


# OobleckDataset does not have any states and ok to use for the entire session.
@pytest.fixture(scope="session", params=["wikitext_dataset", "imagenet_dataset"])
def dataset(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["wikitext_dataset", "imagenet_dataset"])
def dataloaders(request: pytest.FixtureRequest):
    dataset = request.getfixturevalue(request.param)

    training_args = TrainingArguments(
        output_dir="/tmp/output",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
    )

    training_dataloader = OobleckDataLoader(
        dataset,
        training_args,
        LoaderType.Training,
        # total number of microbatches.
        # Currently only have one process, so it should be the same as
        # gradient_accumulation_steps.
        training_args.gradient_accumulation_steps,
        0,
        0,
    )
    eval_dataloader = OobleckDataLoader(
        dataset,
        training_args,
        LoaderType.Evaluation,
        # total number of microbatches.
        # Currently only have one process, so it should be the same as
        # gradient_accumulation_steps.
        training_args.gradient_accumulation_steps,
        0,
        0,
    )
    yield training_dataloader, eval_dataloader


def gpt2_model(wikitext_dataset):
    # Refer to oobleck/examples/*.py for model arguments
    # gpt2-medium
    model_args = {
        "num_hidden_layers": 32,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    }
    return OobleckModel("gpt2", wikitext_dataset.sample, None, "test", model_args)


def resnet_model(imagenet_dataset):
    return OobleckModel(
        "microsoft/resnet-50", imagenet_dataset.sample, None, "test", None
    )


@pytest.fixture(
    scope="session",
    params=[
        (gpt2_model, "wikitext_dataset"),
        (resnet_model, "imagenet_dataset"),
    ],
    ids=["gpt2", "microsoft/resnet-50"],
)
def model(request: pytest.FixtureRequest):
    return request.param[0](request.getfixturevalue(request.param[1]))


@pytest.fixture(
    scope="function",
    params=[
        (gpt2_model, "wikitext_dataset"),
        (resnet_model, "imagenet_dataset"),
    ],
    ids=["gpt2", "microsoft/resnet-50"],
)
def model_function(no_distributed, request: pytest.FixtureRequest):
    return request.param[0](request.getfixturevalue(request.param[1]))


@pytest.fixture(scope="session")
def dummy_profile_results(model: OobleckModel):
    num_layers = len(model.model)
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
        for i in range(1, 17):
            ar_across_nodes[i] = random.random() * 4

        allreduce_across_nodes.append(ar_across_nodes)
        allreduce_in_node.append(
            {1: random.random(), 2: random.random(), 4: random.random()}
        )

    return layers, allreduce_across_nodes, allreduce_in_node


@pytest.fixture(scope="function")
def new_profile_directory(model):
    # This fixture is used to clean up the files created by profile.
    exist = True
    while exist:
        random_tag = "".join(random.choices(string.ascii_letters, k=8))
        path = Path(f"/tmp/oobleck/profiles/{model.model_name}-{random_tag}")
        exist = path.exists()
    yield random_tag
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="function")
def dummy_profile_result_files(
    model: OobleckModel, dummy_profile_results, new_profile_directory
):
    directory = Path(
        f"/tmp/oobleck/profiles/{model.model_name}-{new_profile_directory}"
    )
    directory.mkdir(parents=True, exist_ok=False)

    def _create_files(microbatch_size: int):
        filenames = [
            f"mb{microbatch_size}.json",
            "allreduce_across_nodes.json",
            "allreduce_in_node.json",
        ]
        for filename, result in zip(filenames, dummy_profile_results):
            with directory.joinpath(filename).open(mode="w") as f:
                json.dump(result, f)
                f.flush()

    yield _create_files


@pytest.fixture(scope="session")
def dummy_layer_execution_results(model: OobleckModel, dummy_profile_results):
    layers, allreduce_across_nodes, allreduce_in_node = dummy_profile_results

    results: List[LayerExecutionResult] = []
    for layer, execution, ar_in_node, ar_across_nodes in zip(
        model.model, layers, allreduce_in_node, allreduce_across_nodes
    ):
        results.append(
            LayerExecutionResult(
                layer.index,
                execution["forward"],
                execution["backward"],
                ar_in_node,
                ar_across_nodes,
                execution["mem_required"],
            )
        )
    return LayerExecutionResults(results)


@pytest.fixture(scope="function")
def no_distributed():
    original_env = dict(os.environ)
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    os.environ.pop("LOCAL_RANK", None)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def init_distributed():
    backup = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
    }
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    def _distributed(init_required: bool):
        if init_required:
            if dist.is_initialized():
                return

            dist.init_distributed(dist_backend="nccl", dist_init_required=True)
            assert dist.is_initialized()
        else:
            assert not dist.is_initialized()

    yield _distributed

    if dist.is_initialized():
        dist.destroy_process_group()
        dist.cdb = None
    assert not dist.is_initialized()
    for key, value in backup.items():
        if value is None:
            os.environ.pop(key)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def dummy_pipeline_template(
    model: OobleckModel, dummy_layer_execution_results: LayerExecutionResults
):
    def divide_layers(
        layers: List[LayerExecutionResult], num: int
    ) -> List[List[LayerExecutionResult]]:
        k, m = divmod(len(layers), num)
        return [
            layers[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num)
        ]

    def _create_pipeline_template(num_gpus: int) -> PipelineTemplate:
        layers = divide_layers(dummy_layer_execution_results.get(), num_gpus)
        stages = [
            StageExecutionResult(l, (l[0]._index, l[-1]._index), 1) for l in layers
        ]
        return PipelineTemplate(stages, 0.1, len(model.model), num_gpus, 1)

    return _create_pipeline_template
