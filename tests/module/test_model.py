import pytest

from oobleck.module.model import OobleckModel
from oobleck.module.sharding import get_split_points
from oobleck.execution.dataset import OobleckDataset
from transformers import PretrainedConfig


@pytest.mark.parametrize(
    "model_name,dataset",
    [
        ("gpt2", "wikitext_dataset"),
        ("microsoft/resnet-152", "imagenet_dataset"),
    ],
)
def test_initialize_model(model_name, dataset, request: pytest.FixtureRequest):
    dataset: OobleckDataset = request.getfixturevalue(dataset)
    model = OobleckModel(model_name, dataset.sample, None, "test", None)
    assert isinstance(model, OobleckModel)
    assert model.model_name == model_name
    assert model.model_tag == "test"
    assert model.sample_inputs == dataset.sample
    assert model.training_args is None
    assert isinstance(model.model_args, PretrainedConfig)

    # All split points must be consumed, thus have creating points + 1 sharded layers.
    assert len(model.model) == len(get_split_points(model.model_args)) + 1


@pytest.mark.parametrize(
    "model_name,dataset",
    [
        ("gpt2", "wikitext_dataset"),
        ("microsoft/resnet-152", "imagenet_dataset"),
    ],
)
@pytest.mark.skip(reason="Not implemented yet")
def test_initialize_model_amp(model_name, dataset, request: pytest.FixtureRequest):
    assert False
