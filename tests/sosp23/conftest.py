import logging
import random

import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
)
from oobleck.execution.dataset import OobleckDataset
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
