from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.dataloader import OobleckDataLoader, LoaderType
from oobleck.module.model import OobleckModel
from oobleck.csrc.planning.pipeline_template import PipelineTemplateGenerator
from oobleck.planning.profiler import LayerExecutionResult, get_profile_results

from transformers import TrainingArguments
import pytest
import os
import deepspeed.comm as dist
import random
from typing import List

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@pytest.fixture(scope="session")
def wikitext_dataset():
    return OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")


@pytest.fixture(scope="session")
def imagenet_dataset():
    return OobleckDataset("microsoft/resnet-152", "Maysee/tiny-imagenet")


# OobleckDataset does not have any states and ok to use for the entire session.
@pytest.fixture(scope="session", params=["wikitext_dataset", "imagenet_dataset"])
def dataset(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def dataloaders_function(dataset):
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
    return training_dataloader, eval_dataloader


@pytest.fixture(scope="session", params=["wikitext_dataset", "imagenet_dataset"])
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
    return training_dataloader, eval_dataloader


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
        "microsoft/resnet-152", imagenet_dataset.sample, None, "test", None
    )


@pytest.fixture(
    scope="session",
    params=[
        (gpt2_model, "wikitext_dataset"),
        (resnet_model, "imagenet_dataset"),
    ],
    ids=["gpt2", "microsoft/resnet-152"],
)
def model(request: pytest.FixtureRequest):
    return request.param[0](request.getfixturevalue(request.param[1]))


@pytest.fixture(
    scope="session",
    params=[
        (gpt2_model, "wikitext_dataset"),
        (resnet_model, "imagenet_dataset"),
    ],
    ids=["gpt2", "microsoft/resnet-152"],
)
def model_function(request: pytest.FixtureRequest):
    return request.param[0](request.getfixturevalue(request.param[1]))


@pytest.fixture(scope="module")
def pipeline_template_generator():
    return PipelineTemplateGenerator()


@pytest.fixture(scope="function")
def gpt2_dummy_profile_results(gpt2_model):
    num_layers = len(gpt2_model.model)
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

    results: List[LayerExecutionResult] = []
    for layer, execution, ar_in_node, ar_across_nodes in zip(
        gpt2_model.model, layers, allreduce_in_node, allreduce_across_nodes
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
    return results


@pytest.fixture(scope="session")
def gpt2_profile_results(gpt2_model):
    return get_profile_results(gpt2_model, 1)


@pytest.fixture(scope="function")
def distributed_conf_one():
    backup = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    yield
    for key, value in backup.items():
        if value is None:
            os.environ.pop(key)
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
def distributed():
    dist.init_distributed(dist_backend="nccl")
    assert dist.is_initialized()
    yield
    dist.destroy_process_group()
    dist.cdb = None
    assert not dist.is_initialized()
