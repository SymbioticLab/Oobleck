import copy

import pytest
import torch.fx
from transformers.configuration_utils import PretrainedConfig

from oobleck.execution.layer import init_tensors
from oobleck.module.model import OobleckModel
from oobleck.module.sharding import get_split_points
from tests.conftest import OobleckSingleProcessTestCase


class TestOobleckModel(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def model(self) -> OobleckModel:
        return copy.deepcopy(self.factory.get_model())

    def test_attributes_type(self, model: OobleckModel):
        assert isinstance(model, OobleckModel)
        assert isinstance(model.layers, list)
        # All split points must be consumed, thus have creating points + 1 sharded layers.
        assert len(model.layers) == len(get_split_points(model.model_args)) + 1
        assert model.training_args == self.factory._training_args
        assert isinstance(model.model_args, PretrainedConfig)

        sample_inputs = self.factory.get_dataset().sample
        assert len(model.sample_inputs) == len(sample_inputs)
        for name, sample_input in sample_inputs.items():
            model_sample_input = model.sample_inputs[name]
            assert torch.equal(model_sample_input, sample_input)

    @pytest.mark.skip(reason="AMP not implemented yet")
    def test_model_amp(self):
        pass

    def test_model_layers_type(self, model: OobleckModel):
        assert all(isinstance(layer, torch.fx.GraphModule) for layer in model.layers)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_run_model(self, model: OobleckModel):
        device = torch.device("cuda:0")

        input: tuple[torch.Tensor, ...] = tuple(
            [i.to(device) for i in model.sample_inputs.values()]
        )

        # Because layer now only has metadata, naive copy should fail
        with pytest.raises(NotImplementedError):
            for layer in model.layers:
                layer.to(device)

        # With `init_tensors`, parameters are initialized in GPU
        for layer in model.layers:
            init_tensors(layer, device)
            assert all(p.device == device for p in layer.parameters())

        for layer in model.layers:
            input = layer(*input)

        # output[0]: loss, output[1]: logits
        output = input

        assert isinstance(output, tuple)
        assert isinstance(output[0], torch.Tensor)
        assert output[0].device == device
