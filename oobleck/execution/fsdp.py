from __future__ import annotations

import functools
import logging
from enum import Enum

import torch
import torch.distributed
import torch.fx
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp._runtime_utils import _check_grad_to_accumulate
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy


class StreamType(Enum):
    UNSHARD = "unshard"
    POST_BACKWARD = "post_backward"


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
        # Used for prefetching. Lazily initialited by calling set_prev_and_next_layer().
        self._prev_layer: FullyShardedDataParallelLayer | None
        self._next_layer: FullyShardedDataParallelLayer | None

        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        layer.to(device)
        self._checkpointable = FullyShardedDataParallelLayer.is_checkpointable(layer)
        if self._checkpointable:
            layer = checkpoint_wrapper(layer)

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
        self._param_handle.flat_param.register_hook(
            functools.partial(self._post_backward_hook, self)
        )
        self._param_handle.flat_param._saved_grad_shard: tuple[torch.Tensor] = None

        self._process_group = process_group
        self._streams = streams
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        self.unshard_param_event = torch.cuda.Event()
        # TODO: register pre-backward hooks and post-backward hooks

    def set_prev_and_next_layer(
        self,
        prev_layer: FullyShardedDataParallelLayer | None,
        next_layer: FullyShardedDataParallelLayer | None,
    ):
        self._prev_layer = prev_layer
        self._next_layer = next_layer

    @staticmethod
    def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
        if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
            return False
        if any(
            isinstance(m, torch.nn.CrossEntropyLoss) for _, m in layer.named_modules()
        ):
            return False
        if next(layer.parameters(), None) is None:
            return False
        return True

    def unshard(self, state: HandleTrainingState, need_wait: bool = True):
        """
        Initiate unsharding of parameters (if not in progress).
        Mark all future execution to execute after unsharding to complete if `wait` is True.
        NOTE: it does not synchronize by blocking the current thread.
        """
        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = torch.cuda.current_stream()

        assert state in [
            HandleTrainingState.FORWARD,
            HandleTrainingState.BACKWARD_PRE,
        ], f"Invalid training state {self._param_handle._training_state}"
        self._param_handle._training_state = state

        if self.unshard_param_event.query():
            # Unsharding either not started or already finished.
            if not self._param_handle.needs_unshard():
                # If already finished, return immediately
                return

            with torch.cuda.stream(unshard_stream):
                self._param_handle.pre_unshard()
                self._param_handle.unshard()
                self._param_handle.post_unshard()
                self.unshard_param_event.record(unshard_stream)

        # further execution should wait for unsharding to be done
        if need_wait:
            execution_stream.wait_stream(unshard_stream)

    def reshard(self):
        if self._param_handle.is_sharded(self._param_handle.flat_param):
            return

        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = torch.cuda.current_stream()

        # resharding must be done after execution
        unshard_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(unshard_stream):
            self._param_handle.reshard(True)
            self._param_handle.post_reshard()

        execution_stream.wait_stream(unshard_stream)

    def forward(self, *args) -> tuple[torch.Tensor]:
        self.unshard(HandleTrainingState.FORWARD, True)

        result = self._param_handle._fully_sharded_module(*args)

        # if there is a next layer, prefetch unshard it.
        if self._next_layer is not None:
            self._next_layer.unshard(HandleTrainingState.FORWARD, False)

        self.reshard()
        return result

    @staticmethod
    def _pre_backward_hook(
        self: FullyShardedDataParallelLayer,
        module: torch.nn.Module,
        grad_output: torch.nn.modules.module._grad_t,
    ):
        self.unshard(HandleTrainingState.BACKWARD_PRE, True)
        with torch.cuda.stream(self._streams[StreamType.UNSHARD]):
            self._param_handle.prepare_gradient_for_backward()

    @staticmethod
    def _post_backward_hook(
        self: FullyShardedDataParallelLayer,
        grad_output: torch.Tensor,
    ):
        unshard_stream = self._streams[StreamType.UNSHARD]
        post_backward_stream = torch.cuda.default_stream()
        handle = self._param_handle

        post_backward_stream.wait_stream(torch.cuda.current_stream())
        handle._training_state = HandleTrainingState.BACKWARD_POST

        # follow fsdp._runtime_utils._post_backward_pass()
        # that stores _param_handle.flat_param._saved_grad_shard
        if self._param_handle.uses_sharded_strategy:
            with torch.cuda.stream(post_backward_stream):
                handle.flat_param._post_backward_called = True
                unsharded_grad = grad_output.data
                world_size = torch.distributed.get_world_size(self._process_group)
                chunks = list(unsharded_grad.chunk(world_size))
                new_sharded_grad = torch.empty_like(chunks[0])  # padded

                torch.distributed.reduce_scatter_tensor(
                    output=new_sharded_grad,
                    input=unsharded_grad,
                    group=self._process_group,
                )

                # cast grad dtype to param dtype
                if new_sharded_grad.dtype != handle.flat_param.dtype:
                    new_sharded_grad = new_sharded_grad.to(handle.flat_param.dtype)

                # accumulated gradient if needed
                if handle.flat_param._saved_grad_shard is not None:
                    _check_grad_to_accumulate(
                        new_sharded_grad, handle.flat_param._saved_grad_shard
                    )
                    handle.flat_param._saved_grad_shard += new_sharded_grad
                else:
                    handle.flat_param._saved_grad_shard = new_sharded_grad

                # resharding must wait for post backward
                unshard_stream.wait_stream(post_backward_stream)
                self.reshard()

        # all further execution must be done after sharding
        torch.cuda.current_stream().wait_stream(unshard_stream)

    def backward(self, tensor: torch.Tensor | tuple[tuple[torch.Tensor], torch.Tensor]):
        """
        Stream semantics of backward pass:
        https://pytorch.org/docs/stable/notes/cuda.html
        """

        unshard_stream = self._streams[StreamType.UNSHARD]
        self.unshard(HandleTrainingState.BACKWARD_PRE, True)

        with torch.cuda.stream(unshard_stream):
            self._param_handle.prepare_gradient_for_backward()
            # Tell autograd engine that you need to wait for unsharding to be done
            if isinstance(tensor, torch.Tensor):
                tensor.backward()
            else:
                output, gradients = tensor
                torch.autograd.backward(output, gradients)
        torch.cuda.current_stream().wait_stream(unshard_stream)

        # if there is a previous layer, prefetch unshard it.
        if self._prev_layer is not None:
            self._prev_layer.unshard(HandleTrainingState.BACKWARD_PRE, False)
