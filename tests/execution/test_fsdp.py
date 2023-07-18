import copy

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp.flat_param import HandleShardingStrategy

from oobleck.execution.fsdp import FullyShardedDataParallelLayer
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

    shard_stream = torch.cuda.Stream()
    layers: list[FullyShardedDataParallelLayer] = []

    for pg, layer in zip(pgs, model.layers):
        fsdp_layer = FullyShardedDataParallelLayer(
            layer,
            pg,
            shard_stream,
        )
        fsdp_layer._param_handle.init_flat_param_attributes()
        layers.append(fsdp_layer)

    for prev_layer, layer, next_layer in zip(
        [None] + layers[:-1], layers, layers[1:] + [None]
    ):
        layer.set_prev_and_next_layer(prev_layer, next_layer)

    return layers


def get_layers_and_inputs(
    factory: OobleckStaticClassFactory, dfactory: OobleckDynamicClassFactory
) -> tuple[list[FullyShardedDataParallelLayer], tuple[torch.Tensor]]:
    model = copy.deepcopy(factory.get_model())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    device = torch.device("cuda", torch.cuda.current_device())
    inputs: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )

    # reentrant based activation checkpointing requires
    # input to require grad
    for i in inputs:
        i.requires_grad = i.is_floating_point()

    return fsdp_layers, inputs


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

    fsdp_layers = get_fsdp_layers(fsdp_model, pgs)
    assert len(fsdp_layers) == len(original_model.layers)

    for original_layer, fsdp_layer in zip(original_model.layers, fsdp_layers):
        fsdp_layer.unshard(HandleTrainingState.FORWARD)

        with torch.cuda.stream(
            fsdp_layer._shard_stream
        ), fsdp_layer._param_handle.unflatten_as_params():
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
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)

    for layer in fsdp_layers:
        input = layer(input)

    # input[0]: loss, input[1]: logits
    assert isinstance(input[0], torch.Tensor)


def check_backward(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)

    inputs: list[tuple[torch.Tensor | torch.Size]] = []
    outputs: list[tuple[torch.Tensor | torch.Size]] = []

    torch.cuda.profiler.start()
    with torch.autograd.profiler.emit_nvtx():
        for layer in fsdp_layers:
            new_input = tuple(
                tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor
                for tensor in input
            )
            for ni, i in zip(new_input, input):
                if isinstance(ni, torch.Tensor):
                    ni.requires_grad = i.requires_grad
            inputs.append(new_input)

            need_reshard = bool(layer != fsdp_layers[-1])
            output = layer(new_input, need_reshard)
            outputs.append(output)
            input = output

        # Begin test
        assert isinstance(output, tuple)
        fsdp_layers[-1].backward(output[0])

        # For layers except for the last one,
        # we need to manually get grads of the output tensors
        for index in reversed(range(len(fsdp_layers) - 1)):
            output: list[torch.Tensor] = [
                t
                for t in outputs[index]
                if isinstance(t, torch.Tensor) and t.requires_grad
            ]

            grads: list[torch.nn.Parameter] = [
                t.grad
                for t in inputs[index + 1]
                if isinstance(t, torch.Tensor) and t.requires_grad
            ]

            layer = fsdp_layers[index]
            layer.backward((tuple(output), tuple(grads)))

        torch.cuda.synchronize()

        for layer in fsdp_layers:
            handle = layer._param_handle
            if handle.flat_param.requires_grad:
                assert handle.flat_param.grad is not None
                assert handle.flat_param._saved_grad_shard is not None
            else:
                assert handle.flat_param.grad is None
            assert (
                handle._sharding_strategy == HandleShardingStrategy.NO_SHARD
                or handle.is_sharded(handle.flat_param)
            )
    torch.cuda.profiler.stop()


def check_multiple_forwards_backwards(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    """Test fwd-fwd-bwd-bwd execution order."""
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)

    first_input = input
    second_input = tuple(
        [
            torch.rand_like(
                tensor,
                requires_grad=tensor.requires_grad,
            )
            if tensor.dtype.is_floating_point
            else tensor
            for tensor in input
        ]
    )

    for ni, i in zip(first_input, input):
        if isinstance(ni, torch.Tensor):
            ni.requires_grad = i.requires_grad

    # first fwd
    for layer in fsdp_layers:
        first_input = layer(first_input)

    # second bwd
    for layer in fsdp_layers:
        second_input = layer(second_input)

    first_loss = first_input[0]
    second_loss = second_input[0]

    # Before backward, grad should not be set
    for layer in fsdp_layers:
        assert layer._param_handle.flat_param.grad is None
        assert layer._param_handle.flat_param._saved_grad_shard is None

    torch.cuda.synchronize()

    # First backward
    fsdp_layers[-1].backward(first_loss)
    torch.cuda.synchronize()

    grad_tensors: list[list[torch.nn.Parameter]] = []
    # After first backward, _saved_grad_shard should be set
    for layer in fsdp_layers:
        if layer._param_handle.flat_param.requires_grad:
            assert layer._param_handle.flat_param._saved_grad_shard is not None
            grad_tensors.append(
                layer._param_handle.flat_param._saved_grad_shard.detach().clone()
            )
        else:
            assert layer._param_handle.flat_param._saved_grad_shard is None
            grad_tensors.append(None)

    # Second backward
    fsdp_layers[-1].backward(second_loss)
    torch.cuda.synchronize()

    for grad_tensor, layer in zip(grad_tensors, fsdp_layers):
        # _saved_grad_shard is an accumulated gradient tensor that must be different now.
        second_grad_tensor = layer._param_handle.flat_param._saved_grad_shard
        assert not torch.allclose(second_grad_tensor, grad_tensor)


def check_backward_autograd_execution(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)

    for layer in fsdp_layers:
        input = layer(input)

    output = input
    # Begin test
    assert isinstance(output, tuple)

    # This will automatically calculate the gradients in all layers
    fsdp_layers[-1].backward(output[0])

    torch.cuda.current_stream().synchronize()

    for layer in fsdp_layers:
        handle = layer._param_handle
        if handle.flat_param.requires_grad:
            assert handle.flat_param.grad is not None
        else:
            assert handle.flat_param.grad is None
        assert (
            handle._sharding_strategy == HandleShardingStrategy.NO_SHARD
            or handle.is_sharded(handle.flat_param)
        )


def check_backward_autograd_from_middle(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)

    for layer in fsdp_layers[:-1]:
        input = layer(input)

    previous_output = tuple(
        [t for t in input if isinstance(t, torch.Tensor) and t.requires_grad]
    )

    # Cut torch autograd graph here
    new_input = tuple(
        tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in input
    )
    for ni, i in zip(new_input, input):
        if isinstance(ni, torch.Tensor):
            ni.requires_grad = i.requires_grad

    # Finish execution
    final_output = fsdp_layers[-1](new_input, False)

    torch.cuda.synchronize()

    # Begin test
    assert isinstance(final_output, tuple)
    fsdp_layers[-1].backward(final_output[0])

    # Use backward with last layer's gradient
    grad_tensors: tuple[torch.Tensor] = tuple(
        [t.grad for t in new_input if isinstance(t, torch.Tensor) and t.requires_grad]
    )

    assert len(previous_output) == len(grad_tensors)
    fsdp_layers[-2].backward((previous_output, grad_tensors))

    torch.cuda.current_stream().synchronize()

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
    fsdp_layers, input = get_layers_and_inputs(factory, dfactory)
    params = [l._param_handle.flat_param for l in fsdp_layers]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

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

        output = layer(*new_input)
        inputs.append(new_input)
        outputs.append(output)
        input = output

    torch.cuda.synchronize()

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
        layer.backward((tuple(output), tuple(grads)))

    # Begin test
    # optimizer must not have internal data for now
    for p in optimizer.param_groups[0]["params"]:
        assert len(optimizer.state[p]) == 0

    for l in fsdp_layers:
        l._param_handle.prepare_gradient_for_optim()
    optimizer.step()

    torch.cuda.current_stream().synchronize()

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


@pytest.mark.parametrize(
    "num_gpus",
    [1, 2, 4],
    ids=[
        "1GPU",
        "2GPUs",
        "4GPUs",
    ],
)
class TestFullyShardedDataParallelClass(OobleckMultiProcessTestCase):
    def test_fsdp_unshard(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_unsharded_equal_to_original)

    def test_fsdp_forward(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_forward)

    def test_fsdp_fwd_fwd_bwd_bwd(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_multiple_forwards_backwards)

    def test_fsdp_backward(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_backward)

    def test_fsdp_backward_autograd(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_backward_autograd_execution)

    def test_fsp_backward_in_middle(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_backward_autograd_from_middle)

    def test_fsdp_step(self, num_gpus: int):
        self.run_in_parallel(num_gpus, check_optimizer_step)
