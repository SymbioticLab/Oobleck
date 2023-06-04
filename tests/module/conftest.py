import pytest
from oobleck.module.model import OobleckModel

# Refer to oobleck/examples/*.py for model arguments


@pytest.fixture(scope="session")
def gpt2_model(wikitext_dataset):
    # gpt2-medium
    model_args = {
        "num_hidden_layers": 24,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    }
    return OobleckModel("gpt2", wikitext_dataset.sample, None, "medium", model_args)
