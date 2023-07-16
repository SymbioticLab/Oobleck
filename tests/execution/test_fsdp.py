import copy

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from torch.distributed.fsdp._common_utils import HandleTrainingState

from oobleck.execution.fsdp import FullyShardedDataParallelLayer, StreamType
from oobleck.module.model import OobleckModel
from tests.conftest import (
    GRADIENT_ACCUMULATION_STEP,
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


def get_fsdp_layers(
    model: OobleckModel, pgs: list[torch.distributed.ProcessGroup]
) -> list[FullyShardedDataParallelLayer]:
    assert len(model.layers) == len(pgs)

    unshard_stream = torch.cuda.Stream()
    post_backward_stream = torch.cuda.Stream()
    layers: list[FullyShardedDataParallelLayer] = []

    for pg, layer in zip(pgs, model.layers):
        fsdp_layer = FullyShardedDataParallelLayer(
            layer,
            pg,
            {
                StreamType.UNSHARD: unshard_stream,
                StreamType.POST_BACKWARD: post_backward_stream,
            },
        )
        fsdp_layer._param_handle.init_flat_param_attributes()
        layers.append(fsdp_layer)

    for prev_layer, layer, next_layer in zip(
        [None] + layers[:-1], layers, layers[1:] + [None]
    ):
        layer.set_prev_and_next_layer(prev_layer, next_layer)

    return layers


def check_unsharded_equal_to_original(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    fsdp_model: OobleckModel = copy.deepcopy(factory.get_model())
    original_model: OobleckModel = copy.deepcopy(factory.get_model())
    device = torch.device("cuda", torch.cuda.current_device())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(fsdp_model.layers)

    for pg, layer in zip(pgs, original_model.layers):
        layer.to(device)
        # _sync_module_params_and_buffers(layer, list(layer.parameters()), pg)

    fsdp_layers = get_fsdp_layers(fsdp_model, pgs)
    assert len(fsdp_layers) == len(original_model.layers)

    for original_layer, fsdp_layer in zip(original_model.layers, fsdp_layers):
        fsdp_layer._param_handle._training_state = HandleTrainingState.FORWARD
        fsdp_layer.unshard(HandleTrainingState.FORWARD)

        with fsdp_layer._param_handle.unflatten_as_params():
            original_params = list(original_layer.parameters())
            fsdp_params = list(
                fsdp_layer._param_handle._fully_sharded_module.parameters()
            )
            assert len(original_params) == len(fsdp_params)
            assert all(
                torch.allclose(o, f) for o, f in zip(original_params, fsdp_params)
            )


def check_forward(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    model = copy.deepcopy(factory.get_model())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )

    for layer in fsdp_layers:
        layer._param_handle._training_state = HandleTrainingState.FORWARD
        input = layer(*input)

    # input[0]: loss, input[1]: logits
    assert isinstance(input[0], torch.Tensor)


def check_backward(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    model = copy.deepcopy(factory.get_model())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )
    # reentrant based activation checkpointing requires
    # input to require grad
    for i in input:
        i.requires_grad = i.is_floating_point()

    inputs: list[tuple[torch.Tensor | torch.Size]] = []
    outputs: list[tuple[torch.Tensor | torch.Size]] = []

    for layer in fsdp_layers:
        new_input = tuple(
            tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in input
        )
        for ni, i in zip(new_input, input):
            if isinstance(ni, torch.Tensor):
                ni.requires_grad = i.requires_grad
        inputs.append(new_input)

        layer._param_handle._training_state = HandleTrainingState.FORWARD
        output = layer(*new_input)
        outputs.append(output)
        input = output

    torch.cuda.synchronize()

    # Begin test
    assert isinstance(output, tuple)
    fsdp_layers[-1]._param_handle._training_state = HandleTrainingState.BACKWARD_PRE
    fsdp_layers[-1].backward(output[0])

    # For layers except for the last one,
    # we need to manually get grads of the output tensors
    for index in reversed(range(len(fsdp_layers) - 1)):
        output: list[torch.Tensor] = [
            t for t in outputs[index] if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        grads: list[torch.nn.Parameter] = [
            t.grad
            for t in inputs[index + 1]
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        layer = fsdp_layers[index]
        layer._param_handle._training_state = HandleTrainingState.BACKWARD_PRE
        layer.backward((tuple(output), tuple(grads)))

    torch.cuda.synchronize()

    for layer in fsdp_layers:
        handle = layer._param_handle
        if handle.flat_param.requires_grad:
            assert handle.flat_param.grad is not None
        else:
            assert handle.flat_param.grad is None
        assert handle.is_sharded(handle.flat_param)


def check_optimizer_step(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    """
    Oobleck has a single optimizer that contains all parameters from
    several layers. Because each layer doesn't have its own optimizer,
    here we emulate the optimizer.
    """
    model = copy.deepcopy(factory.get_model())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    params = [l._param_handle.flat_param for l in fsdp_layers]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )
    # reentrant based activation checkpointing requires
    # input to require grad
    for i in input:
        if isinstance(i, torch.Tensor):
            i.requires_grad = i.is_floating_point()

    inputs: list[tuple[torch.Tensor | torch.Size]] = []
    outputs: list[tuple[torch.Tensor | torch.Size]] = []

    for layer in fsdp_layers:
        new_input = tuple(
            tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in input
        )
        for ni, i in zip(new_input, input):
            if isinstance(ni, torch.Tensor):
                ni.requires_grad = i.requires_grad

        layer._param_handle._training_state = HandleTrainingState.FORWARD
        output = layer(*new_input)
        inputs.append(new_input)
        outputs.append(output)
        input = output

    fsdp_layers[-1]._param_handle._training_state = HandleTrainingState.BACKWARD_PRE
    fsdp_layers[-1].backward(input[0])
    for index in reversed(range(len(fsdp_layers) - 1)):
        output: list[torch.Tensor] = [
            t for t in outputs[index] if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        grads: list[torch.nn.Parameter] = [
            t.grad
            for t in inputs[index + 1]
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        layer = fsdp_layers[index]
        layer._param_handle._training_state = HandleTrainingState.BACKWARD_PRE
        layer.backward((tuple(output), tuple(grads)))

    # Begin test
    # optimizer must not have internal data for now
    for p in optimizer.param_groups[0]["params"]:
        assert len(optimizer.state[p]) == 0

    for l in fsdp_layers:
        l._param_handle.prepare_gradient_for_optim()
    optimizer.step()

    torch.cuda.synchronize()

    # check parameters are still sharded
    assert all(
        l._param_handle.is_sharded(l._param_handle.flat_param) for l in fsdp_layers
    )

    # optimizer must have internal data for now
    assert len(fsdp_layers) == len(optimizer.param_groups[0]["params"])
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


class TestFullyShardedDataParallelClass(OobleckMultiProcessTestCase):
    def test_fsdp_unshard(self):
        self.run_in_parallel(2, check_unsharded_equal_to_original)

    def test_fsdp_forward(self):
        self.run_in_parallel(2, check_forward)

    def test_fsdp_backward(self):
        self.run_in_parallel(2, check_backward)

    def test_fsdp_step(self):
        self.run_in_parallel(2, check_optimizer_step)
