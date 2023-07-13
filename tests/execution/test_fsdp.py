import deepspeed.comm as dist
import pytest
import torch

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

    for layer in layers:
        layer.unshard(False)

    unshard_stream.synchronize()

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
