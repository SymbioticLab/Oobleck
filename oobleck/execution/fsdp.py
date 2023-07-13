from enum import Enum

import torch
import torch.distributed
import torch.fx
from torch.distributed import ProcessGroup
from torch.distributed.fsdp.flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
)

from oobleck.module.layer import Layer


class ShardStatus(Enum):
    NotExist = 0
    Sharded = 1
    Unsharded = 2


class FullyShardedDataParallelLayer(torch.nn.Module):
    """
    Copy parameters in `layer` to CUDA device, then flatten and shard it.
    In GPU memory, there is only a `FlatParameter` object.

    After initialization, this layer always has sharded parameters.
    During `forward()`, it unshards the parameter, use it, and reshards it.
    For `backward()`, it breaks torch.autograd rule and manually accepts gradients
    and feed it into `torch.autograd.backward()` after unsharding.
    """

    def __init__(
        self,
        layer: torch.fx.GraphModule,
        process_group: ProcessGroup,
        streams: dict[str, torch.cuda.Stream],
    ):
        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        layer.to(device)
        self._param_handle = FlatParamHandle(
            params=layer.parameters(),
            fully_sharded_module=layer,
            device=device,
            sharding_strategy=HandleShardingStrategy.FULL_SHARD,
            offload_params=False,
            mp_param_dtype=torch.float32,  # TODO: change to bf16
            mp_reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
            process_group=process_group,
            use_orig_params=True,
        )

        # self.shard()
        self._param_handle.init_flat_param_attributes()

        self._process_group = process_group
        self._streams = streams
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        self._parameter_shard: ShardStatus = ShardStatus.Unsharded
        self._gradients_shard: ShardStatus = ShardStatus.NotExist

    def forward(self, *args) -> tuple[torch.Tensor]:
        return self._param_handle._fully_sharded_module(*args)

    def backward(self, output: tuple[torch.Tensor], gradients: tuple[torch.Tensor]):
        pass


class OobleckFullyShardedDataParallel:
    """A variant of Deepspeed Zero 3 or FSDP for Oobleck 3D parallelism.
    `TrainingState` based PyTorch FullyShardedDataParallel implementation is not suitable to
    integrate FSDP with pipeline parallelism, where multiple forwards and backwards are executed
    consecutively, not potentially interleaved.
    Instead, Oobleck only manages its state as [parameter/gradient]_[sharded/unsharded].

    Instead, Oobleck implementation provides a FSDPStage object, which is a list of layers.
    Each stage provides explicitly modularized functions that are executed in some order that works
    in conjunction with pipeline parallelism.
    """

    def __init__(self, layers: list[FullyShardedDataParallelLayer]):
        self._layers = layers
        self._streams: dict[str, torch.cuda.Stream] = {
            # stream for default computation
            "default": torch.cuda.current_stream(),
            # stream for unsharding parameters in forward pass
            "forward": torch.cuda.Stream(),
            # stream for unsharding gradients in backward pass
            "backward": torch.cuda.Stream(),
        }
