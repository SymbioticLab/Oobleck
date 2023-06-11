import pytest
import torch.fx
from transformers import PretrainedConfig

from oobleck.execution.dataset import OobleckDataset
from oobleck.module.layer import Layer
from oobleck.module.model import OobleckModel
from oobleck.module.sharding import get_split_points


@pytest.mark.parametrize(
    "model_name,dataset",
    [
        ("gpt2", "wikitext_dataset"),
        ("microsoft/resnet-50", "imagenet_dataset"),
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
        ("microsoft/resnet-50", "imagenet_dataset"),
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


def test_model_layers_param_on_cpu(model_function: OobleckModel):
    for layer in model_function.model:
        assert all(p.device == torch.device("cpu") for p in layer.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_model(
    device: str,
    model_function: OobleckModel,
):
    pytest.mark.skipif(
        device.startswith("cuda") and not torch.cuda.is_available(),
        reason="No GPU available",
    )

    device = torch.device(device)

    input = tuple([i.to(device) for i in model_function.sample_inputs.values()])
    for layer in model_function.model:
        layer.to(device)
        assert all(p.device == device for p in layer.parameters())

    for layer in model_function.model:
        input = layer(*input)

    assert "loss" in input
    assert isinstance(input["loss"], torch.Tensor)
    assert input["loss"].device == device
