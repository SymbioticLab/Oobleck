import logging
from enum import Enum

import torch
import torch.distributed
import torch.fx
from torch.distributed import ProcessGroup
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp._init_utils import _sync_module_params_and_buffers
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

        _sync_module_params_and_buffers(layer, list(layer.parameters()), process_group)

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
            use_orig_params=False,
        )
        self._param_handle.shard()

        self._process_group = process_group
        self._streams = streams
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        self.unshard_param_event = torch.cuda.Event()
        # TODO: register pre-backward hooks and post-backward hooks

    def unshard(self):
        """
        Initiate unsharding of parameters (if not in progress).
        Mark all future execution to execute after unsharding to complete if `wait` is True.
        NOTE: it does not synchronize by blocking the current thread.
        """
        if self.unshard_param_event.query():
            # Unsharding either not started or already finished.
            if not self._param_handle.needs_unshard():
                # If already finished, return immediately
                return

            # If we don't have an event, it means there is no unsharding
            # in process. Iniiate unsharing

            assert self._param_handle._training_state in [
                HandleTrainingState.FORWARD,
                HandleTrainingState.BACKWARD_PRE,
            ]

            unshard_stream = self._streams[StreamType.UNSHARD]
            execution_stream = self._streams[StreamType.EXECUTION]

            with torch.cuda.stream(unshard_stream):
                self._param_handle.pre_unshard()
                self._param_handle.unshard()
                self._param_handle.post_unshard()
                self.unshard_param_event.record(unshard_stream)

            # further execution should wait for unsharding to be done
            execution_stream.wait_stream(unshard_stream)

    def reshard(self):
        if self._param_handle.is_sharded(self._param_handle.flat_param):
            return

        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = self._streams[StreamType.EXECUTION]

        # resharding must be done after execution
        unshard_stream.wait_stream(execution_stream)

        self._param_handle.reshard(True)
        self._param_handle.post_reshard()

    def unshard_grad(self):
        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = self._streams[StreamType.EXECUTION]

        with torch.cuda.stream(unshard_stream):
            self._param_handle.unshard_grad()

        # further execution should wait for unsharding to be done
        execution_stream.wait_stream(unshard_stream)

    def reshard_grad(self):
        # resharding must be done after execution
        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = self._streams[StreamType.EXECUTION]
        unshard_stream.wait_stream(execution_stream)

        self._param_handle.reshard_grad()

    def forward(self, *args) -> tuple[torch.Tensor]:
        self._param_handle._training_state = HandleTrainingState.FORWARD
        self.unshard()
        result = self._param_handle._fully_sharded_module(*args)
        self.reshard()
        self._param_handle._training_state = HandleTrainingState.IDLE
        return result

    def backward(self, tensor: torch.Tensor | tuple[tuple[torch.Tensor], torch.Tensor]):
        self._param_handle._training_state = HandleTrainingState.BACKWARD_PRE
        self.unshard()
        self.unshard_param_event.synchronize()
        self._param_handle.prepare_gradient_for_backward()
        if isinstance(tensor, torch.Tensor):
            tensor.backward()
        else:
            output, gradients = tensor
            torch.autograd.backward(output, gradients)
        self.reshard()
        torch.cuda.synchronize()
        # TODO: what to do after backward?


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
