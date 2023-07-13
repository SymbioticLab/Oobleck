import deepspeed.comm as dist
import pytest
import torch
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleTrainingState

from oobleck.execution.fsdp import FullyShardedDataParallelLayer
from tests.conftest import (
    GRADIENT_ACCUMULATION_STEP,
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


def fsdplayer(
    factory: OobleckStaticClassFactory,
    dfactory: OobleckDynamicClassFactory,
):
    pg = dist.new_group(ranks=dfactory._ranks)
    model = factory.get_model()

    layers: list[FullyShardedDataParallelLayer] = []
    for layer in model.layers:
        layers.append(FullyShardedDataParallelLayer(layer, pg, {}))

    print("Before unsharding memory: ", torch.cuda.memory_allocated())

    for layer in layers:
        layer._param_handle.pre_unshard()
        layer._param_handle.unshard()
        layer._param_handle.post_unshard()

    torch.cuda.synchronize()
    print("After unsharding memory: ", torch.cuda.memory_allocated())

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [i.to(device) for i in model.sample_inputs.values()]
    )
    for layer in layers:
        input = layer(*input)

    assert "loss" in input
    assert isinstance(input["loss"], torch.Tensor)


class TestFullyShardedDataParallelClass(OobleckMultiProcessTestCase):
    def test_fsdplayer(self):
        self.run_in_parallel(2, fsdplayer)
