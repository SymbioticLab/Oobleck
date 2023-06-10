import pytest
import torch.fx

from oobleck.module.model import OobleckModel
from oobleck.module.layer import Layer
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
def test_initialize_model(
    model_name: str, dataset: str, request: pytest.FixtureRequest
):
    dataset: OobleckDataset = request.getfixturevalue(dataset)
    model = OobleckModel(model_name, dataset.sample, None, "test", None)
    assert isinstance(model, OobleckModel)
    assert model.model_name == model_name
    assert model.model_tag == "test"
    assert model.sample_inputs == dataset.sample
    assert model.training_args is None
    assert isinstance(model.model_args, PretrainedConfig)

    assert isinstance(model.model, list)
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
def test_initialize_model_amp(
    model_name: str, dataset: str, request: pytest.FixtureRequest
):
    assert False


def test_model_layers_type(model: OobleckModel):
    for index, layer in enumerate(model.model):
        assert isinstance(layer, Layer)
        assert isinstance(layer.layer, torch.fx.GraphModule)
        assert layer.index == index


def test_model_layers_param_on_cpu(model: OobleckModel):
    for layer in model.model:
        assert all(p.device == torch.device("cpu") for p in layer.parameters())


def test_run_model_on_cpu(model: OobleckModel):
    input = tuple(model.sample_inputs.values())
    for layer in model.model:
        input = layer(*input)
    assert "loss" in input
    assert isinstance(input["loss"], torch.Tensor)
