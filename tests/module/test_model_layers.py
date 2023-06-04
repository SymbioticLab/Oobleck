import torch
import torch.fx
import pytest

from oobleck.module.model import OobleckModel
from oobleck.module.layer import Layer
from oobleck.module.sharding import get_split_points


def test_gpt_model(gpt2_model):
    assert isinstance(gpt2_model, OobleckModel)
    assert gpt2_model.model_name == "gpt2"
    assert gpt2_model.model_tag == "test"


def test_layers(gpt2_model):
    assert isinstance(gpt2_model.model, list)
    assert len(gpt2_model.model) == len(get_split_points(gpt2_model.model_args)) + 1
    for index, layer in enumerate(gpt2_model.model):
        assert isinstance(layer, Layer)
        assert layer.index == index


def test_layers_param_on_cpu(gpt2_model):
    for layer in gpt2_model.model:
        assert all(p.device == torch.device("cpu") for p in layer.parameters())


def test_layer_fx(gpt2_model):
    for layer in gpt2_model.model:
        assert isinstance(layer.layer, torch.fx.GraphModule)
