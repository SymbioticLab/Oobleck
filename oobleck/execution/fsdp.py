from enum import Enum

import torch
import torch.distributed
import torch.fx
from torch.distributed import ProcessGroup
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy


class StreamType(Enum):
    UNSHARD = "unshard"
    UNSHARD_GRAD = "unshard_grad"
    EXECUTION = "execution"


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
        streams: dict[StreamType, torch.cuda.Stream],
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

        self._param_handle.init_flat_param_attributes()
        self._param_handle._check_on_compute_device(self._param_handle.flat_param)

        self._process_group = process_group
        self._streams = streams
        self._unshard_param_event: torch.cuda.Event | None = None
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        # TODO: register pre-backward hooks and post-backward hooks

    def unshard(self, wait: bool = False):
        """
        Initiate unsharding of parameters (if not in progress).
        Wait for unsharding to complete if `wait` is True.
        """
        if not self._param_handle.is_sharded(self._param_handle.flat_param):
            # Return immediately if the parameter is not sharded
            return

        if self._unshard_param_event is None:
            # If we don't have an event, it means there is no unsharding
            # in process. Iniiate unsharing
            with self._streams[StreamType.UNSHARD]:
                self._param_handle.pre_unshard()
                self._param_handle.unshard()
                self._param_handle.post_unshard()
                self._unshard_param_event = torch.cuda.Event()
                self._unshard_param_event.record()

        if wait:
            self._unshard_param_event.synchronize()
            self._unshard_param_event = None

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
