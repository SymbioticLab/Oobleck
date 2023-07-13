import time

import deepspeed.comm as dist
import pytest
import torch
from torch.distributed.fsdp._common_utils import HandleTrainingState

from oobleck.execution.fsdp import FullyShardedDataParallelLayer, StreamType
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
    unshard_stream = torch.cuda.Stream()
    for layer in model.layers:
        layers.append(
            FullyShardedDataParallelLayer(
                layer, pg, {StreamType.UNSHARD: unshard_stream}
            )
        )

    start = time.time_ns()
    for layer in layers:
        layer._param_handle.init_flat_param_attributes()
        layer.unshard()
    end = time.time_ns()
    print(f"Unsharding took {(end - start) / 1_000} microseconds")

    unshard_stream.synchronize()

    device = torch.device("cuda", torch.cuda.current_device())
    input: tuple[torch.Tensor, ...] = tuple(
        [i.to(device) for i in model.sample_inputs.values()]
    )
    for layer in layers:
        with layer._param_handle.unflatten_as_params():
            input = layer(*input)

    assert "loss" in input
    assert isinstance(input["loss"], torch.Tensor)

    start = time.time_ns()
    for layer in layers:
        layer.reshard()
    end = time.time_ns()
    print(f"Resharding took {(end - start) / 1_000} microseconds")

    unshard_stream.synchronize()

    # # ========================
    # for i in range(3):
    #     print("range: ", i)

    #     layers[0].unshard()
    #     torch.cuda.synchronize()

    #     layers[0].reshard()
    #     torch.cuda.synchronize()
    # # ========================


class TestFullyShardedDataParallelClass(OobleckMultiProcessTestCase):
    def test_fsdplayer(self):
        self.run_in_parallel(2, fsdplayer)
