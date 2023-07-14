import copy

import deepspeed.comm as dist
import pytest
import torch
from torch.distributed import ProcessGroup
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp._init_utils import _sync_module_params_and_buffers

from oobleck.execution.fsdp import FullyShardedDataParallelLayer, StreamType
from oobleck.module.model import OobleckModel
from tests.conftest import (
    GRADIENT_ACCUMULATION_STEP,
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


def get_fsdp_layers(
    model: OobleckModel, pgs: list[ProcessGroup]
) -> list[FullyShardedDataParallelLayer]:
    assert len(model.layers) == len(pgs)

    unshard_stream = torch.cuda.Stream()
    layers: list[FullyShardedDataParallelLayer] = []

    for pg, layer in zip(pgs, model.layers):
        fsdp_layer = FullyShardedDataParallelLayer(
            layer,
            pg,
            {
                StreamType.UNSHARD: unshard_stream,
                StreamType.EXECUTION: torch.cuda.default_stream(),
            },
        )
        fsdp_layer._param_handle.init_flat_param_attributes()
        layers.append(fsdp_layer)

    return layers


def check_unsharded_equal_to_original(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    model = factory.get_model()
    fsdp_model: OobleckModel = copy.deepcopy(model)
    original_model: OobleckModel = copy.deepcopy(model)
    device = torch.device("cuda", torch.cuda.current_device())
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    for pg, layer in zip(pgs, original_model.layers):
        layer.to(device)
        _sync_module_params_and_buffers(layer, list(layer.parameters()), pg)

    fsdp_layers = get_fsdp_layers(fsdp_model, pgs)
    assert len(fsdp_layers) == len(original_model.layers)

    for original_layer, fsdp_layer in zip(original_model.layers, fsdp_layers):
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
    model = factory.get_model()
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )

    for layer in fsdp_layers:
        input = layer(*input)

    assert "loss" in input
    assert isinstance(input["loss"], torch.Tensor)


def check_backward(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    model = factory.get_model()
    pg = dist.new_group(ranks=dfactory._ranks)
    pgs = [pg] * len(model.layers)

    fsdp_layers = get_fsdp_layers(model, pgs)

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [tensor.to(device) for tensor in model.sample_inputs.values()]
    )

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

    # Begin test
    assert "loss" in input
    fsdp_layers[-1].backward(input["loss"])

    for index in reversed(range(len(fsdp_layers) - 1)):
        output = [
            t for t in outputs[index] if isinstance(t, torch.Tensor) and t.requires_grad
        ]
        layer = fsdp_layers[index]
        next_input = [
            t.grad
            for t in inputs[index + 1]
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        print(f"Backward for {index}th layer")
        layer.backward((tuple(output), tuple(next_input)))

    # TODO: what to assert here?


class TestFullyShardedDataParallelClass(OobleckMultiProcessTestCase):
    def test_fsdp_unshard(self):
        self.run_in_parallel(2, check_unsharded_equal_to_original)

    def test_fsdp_forward(self):
        self.run_in_parallel(2, check_forward)

    def test_fsdp_backward(self):
        self.run_in_parallel(2, check_backward)
