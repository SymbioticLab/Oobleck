import pytest
import torch.fx
from transformers.configuration_utils import PretrainedConfig

from oobleck.module.layer import Layer
from oobleck.module.model import OobleckModel
from oobleck.module.sharding import get_split_points
from tests.conftest import OobleckSingleProcessTestCase


class TestOobleckModel(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def model(self) -> OobleckModel:
        return self.factory.get_model()

    def test_attributes_type(self, model: OobleckModel):
        assert isinstance(model, OobleckModel)
        assert isinstance(model.layers, list)
        # All split points must be consumed, thus have creating points + 1 sharded layers.
        assert len(model.layers) == len(get_split_points(model.model_args)) + 1
        assert model.training_args == self.factory._training_args
        assert isinstance(model.model_args, PretrainedConfig)
        assert model.sample_inputs == self.factory.get_dataset().sample

    @pytest.mark.skip(reason="AMP not implemented yet")
    def test_model_amp(self):
        pass

    def test_model_layers_type(self, model: OobleckModel):
        for index, layer in enumerate(model.layers):
            assert isinstance(layer, Layer)
            assert isinstance(layer.layer, torch.fx.GraphModule)
            assert layer.index == index

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_run_model(self, model: OobleckModel):
        device = torch.device("cuda:0")

        input: tuple[torch.Tensor, ...] = tuple(
            [i.to(device) for i in model.sample_inputs.values()]
        )
        for layer in model.layers:
            layer.to(device)
            assert all(p.device == device for p in layer.parameters())

        for layer in model.layers:
            input = layer(*input)

        assert "loss" in input
        assert isinstance(input["loss"], torch.Tensor)
        assert input["loss"].device == device
