import csv
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoModelForPreTraining, PreTrainedModel

from oobleck.profiler import ModelProfiler


@pytest.fixture
def model() -> PreTrainedModel:
    config = AutoConfig.from_pretrained("gpt2")
    model: PreTrainedModel = AutoModelForPreTraining.from_config(config)
    model.gradient_checkpointing_enable()
    return model


def test_profile_model(tmp_path: Path, model: PreTrainedModel):
    layers = (
        ["transformer.wte", "transformer.wpe", "transformer.drop"]
        + [f"transformer.h.{i}" for i in range(model.config.n_layer)]
        + ["transformer.ln_f", "lm_head"]
    )

    profiler = ModelProfiler("gpt2-test", model, layers, tmp_path)
    assert profiler.profile_exists() is False

    batch = model.dummy_inputs
    batch["labels"] = batch["input_ids"]
    profiler.profile(batch)

    assert profiler.profile_exists() is True
    with open(profiler.profile_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert [row["layer_name"] for row in rows] == layers
