import copy
import itertools
import time
from unittest.mock import patch

import pytest
import torch.distributed
from torch.distributed.fsdp._common_utils import HandleTrainingState

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
    shard_stream = torch.cuda.Stream()
    for layer_id, (pg, layer) in enumerate(zip(pgs, model.layers)):
        layers.append(Layer(layer_id, layer, pg, shard_stream))

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
            for param in layer._param_handle._fully_sharded_module.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                else:
                    assert param.grad is None

    @staticmethod
    def step(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        params = itertools.chain.from_iterable(
            [l._param_handle._fully_sharded_module.parameters() for l in layers]
        )
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


class TestShardedLayer(OobleckMultiProcessTestCase):
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

        torch.cuda.synchronize()
        # input[0]: loss, input[1]: logits
        assert isinstance(input[0], torch.Tensor)

        # Check all layers are sharded
        for layer in layers:
            assert layer._param_handle.needs_unshard()

    @staticmethod
    def backward(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        layers[0].unshard_params(HandleTrainingState.FORWARD)

        output: tuple[torch.Tensor]
        for layer in layers:
            output = layer(input)
            input = output

        # grad must be None before executing backward
        for layer in layers:
            assert layer._param_handle.flat_param.grad is None

        # Begin test
        layers[-1].backward(output[0])
        torch.cuda.synchronize()

        for layer in layers:
            # flat_param.grad must be cleared
            assert layer._param_handle.flat_param.grad is None

            # Instead, grad must be stored in _saved_grad_shard
            assert layer._param_handle.flat_param._saved_grad_shard is not None

            # grad shape must be equal to the sharded size
            assert (
                layer._param_handle.flat_param._sharded_size
                == layer._param_handle.flat_param._saved_grad_shard.shape
            )

            layer.remove_post_backward_hooks()

    @staticmethod
    def step(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
    ):
        layers, input = TestNoshardedLayer.get_layers_and_inputs(
            factory, torch.distributed.group.WORLD
        )

        params = itertools.chain.from_iterable(
            [l._param_handle._fully_sharded_module.parameters() for l in layers]
        )
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

    # @pytest.mark.parametrize("function", ["forward", "backward", "step"])
    @pytest.mark.parametrize("function", ["forward", "backward", "step"])
    def test_layer(self, function: str):
        func = getattr(self, function)
        self.run_in_parallel(4, func)
