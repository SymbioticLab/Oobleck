import copy

import pytest
import torch.distributed

from oobleck.execution.layer import Layer
from oobleck.module.model import OobleckModel
from tests.conftest import (
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


def get_layers(model: OobleckModel, pgs: list[torch.distributed.ProcessGroup]):
    assert len(model.layers) == len(pgs)

    layers: list[Layer] = []
    for index, (pg, layer) in enumerate(zip(pgs, model.layers)):
        layers.append(Layer(index, layer, pg))

    return layers


class TestNoshardedLayer(OobleckMultiProcessTestCase):
    @staticmethod
    def get_layers_and_inputs(
        factory: OobleckStaticClassFactory,
        process_group: torch.distributed.ProcessGroup,
    ) -> tuple[list[Layer], tuple[torch.Tensor]]:
        model = copy.deepcopy(factory.get_model())
        pgs = [process_group] * len(model.layers)

        layers = get_layers(model, pgs)

        device = torch.device("cuda", torch.cuda.current_device())
        inputs: tuple[torch.Tensor, ...] = tuple(
            [tensor.to(device) for tensor in model.sample_inputs.values()]
        )

        # reentrant based activation checkpointing requires
        # input to require grad
        for i in inputs:
            i.requires_grad = i.is_floating_point()

        return layers, inputs

    @staticmethod
    def forward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        output: tuple[torch.Tensor]
        for layer in layers:
            output = layer(input)
            input = output

        # input[0]: loss, input[1]: logits
        assert isinstance(input[0], torch.Tensor)

    @staticmethod
    def backward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        output: tuple[torch.Tensor]
        for layer in layers:
            output = layer(input)
            input = output

        # grad must be None before executing backward
        assert all([l._param_handle.flat_param.grad is None for l in layers])

        # Begin test
        output[0].backward()

        # grad must be set after executing backward
        for layer in layers:
            if layer._param_handle.flat_param.requires_grad:
                assert layer._param_handle.flat_param.grad is not None
            else:
                assert layer._param_handle.flat_param.grad is None

    @staticmethod
    def step(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        params = [l._param_handle.flat_param for l in layers]
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        output: tuple[torch.Tensor]
        for layer in layers:
            output = layer(input)
            input = output
        output[0].backward()

        # Begin test
        # optimizer must not have internal data
        for p in optimizer.param_groups[0]["params"]:
            assert len(optimizer.state[p]) == 0

        for l in layers:
            l._param_handle.prepare_gradient_for_optim()
        optimizer.step()

        # optimizer must have internal data for now
        assert len(layers) == len(optimizer.param_groups[0]["params"])
        p: torch.Tensor
        for p in optimizer.param_groups[0]["params"]:
            # If FSDP is used, some too small tensors might be only on rank 0,
            # thus pass if size is 0.
            if p.numel() == 0:
                continue
            assert all(
                key in optimizer.state[p] for key in ["step", "exp_avg", "exp_avg_sq"]
            )
            assert p.shape == optimizer.state[p]["exp_avg"].shape
            assert p.shape == optimizer.state[p]["exp_avg_sq"].shape

    @pytest.mark.parametrize("function", ["forward", "backward", "step"])
    def test_layer(self, function: str):
        func = getattr(self, function)
        self.run_in_parallel(1, func)
