from __future__ import annotations

import functools

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
        shard_stream: torch.cuda.Stream,
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

        self._process_group = process_group
        self._shard_stream = shard_stream
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        self._param_handle = FlatParamHandle(
            params=layer.parameters(),
            fully_sharded_module=layer,
            device=device,
            sharding_strategy=HandleShardingStrategy.FULL_SHARD
            if self._group_size > 1
            else HandleShardingStrategy.NO_SHARD,
            offload_params=False,
            mp_param_dtype=torch.float32,  # TODO: change to bf16
            mp_reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
            process_group=process_group,
            use_orig_params=False,
        )
        self._param_handle.shard()
        self._param_handle.flat_param._saved_grad_shard: tuple[torch.Tensor] = None
        self._param_handle.flat_param.register_hook(
            functools.partial(self._post_backward_hook, self)
        )

        self._unshard_param_event = torch.cuda.Event()
        self._reshard_param_event = torch.cuda.Event()
        self._forward_event = torch.cuda.Event()
        self._backward_event = torch.cuda.Event()

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

    def unshard(self, state: HandleTrainingState):
        """
        Initiate unsharding of parameters (if not in progress).
        Mark all future execution to execute after unsharding to complete if `wait` is True.
        NOTE: it does not synchronize by blocking the current thread.
        """
        if self._param_handle._sharding_strategy == HandleShardingStrategy.NO_SHARD:
            return

        assert state in [
            HandleTrainingState.FORWARD,
            HandleTrainingState.BACKWARD_PRE,
        ], f"Invalid training state {self._param_handle._training_state}"
        self._param_handle._training_state = state

        if self._unshard_param_event.query():
            # Unsharding either not started or already finished.
            if (
                not self._param_handle.needs_unshard()
                and state is not HandleTrainingState.BACKWARD_PRE
            ):
                return

            self._shard_stream.wait_event(self._reshard_param_event)
            with torch.cuda.stream(self._shard_stream):
                self._param_handle.pre_unshard()
                self._param_handle.unshard()
                self._param_handle.post_unshard()
                self._unshard_param_event.record(self._shard_stream)

    def reshard(self):
        if self._param_handle._sharding_strategy == HandleShardingStrategy.NO_SHARD:
            return
        if self._param_handle.is_sharded(self._param_handle.flat_param):
            return

        self._shard_stream.wait_event(self._forward_event)
        with torch.cuda.stream(self._shard_stream):
            self._param_handle.reshard(True)
            self._param_handle.post_reshard()
            self._reshard_param_event.record(self._shard_stream)

    def forward(
        self, inputs: tuple[torch.Tensor], reshard_after: bool = True
    ) -> tuple[torch.Tensor]:
        self.unshard(HandleTrainingState.FORWARD)
        # wait for unshard event to complete
        torch.cuda.current_stream().wait_event(self._unshard_param_event)

        # if there is a next layer, prefetch unshard it.
        if self._next_layer is not None:
            self._next_layer.unshard(HandleTrainingState.FORWARD)

        result = self._param_handle._fully_sharded_module(*inputs)
        self._forward_event.record(torch.cuda.current_stream())

        if reshard_after:
            self.reshard()

        return result

    @staticmethod
    def _post_backward_hook(
        self: FullyShardedDataParallelLayer,
        grad_output: torch.Tensor,
    ):
        handle = self._param_handle

        unsharded_grad = grad_output.data
        # follow fsdp._runtime_utils._post_backward_pass()
        # that stores _param_handle.flat_param._saved_grad_shard
        # self._shard_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(self._shard_stream):
        #     if self._param_handle.uses_sharded_strategy:
        #         world_size = torch.distributed.get_world_size(self._process_group)
        #         chunks = list(unsharded_grad.chunk(world_size))
        #         new_sharded_grad = torch.empty_like(chunks[0])  # padded

        #         torch.distributed.reduce_scatter_tensor(
        #             output=new_sharded_grad,
        #             input=unsharded_grad,
        #             group=self._process_group,
        #         )
        #     else:
        #         new_sharded_grad = unsharded_grad
        # torch.cuda.current_stream().wait_stream(self._shard_stream)
        new_sharded_grad = unsharded_grad

        # cast grad dtype to param dtype
        if new_sharded_grad.dtype != handle.flat_param.dtype:
            new_sharded_grad = new_sharded_grad.to(
                handle.flat_param.dtype, non_blocking=True
            )

        # accumulated gradient if needed
        if handle.flat_param._saved_grad_shard is not None:
            _check_grad_to_accumulate(
                new_sharded_grad, handle.flat_param._saved_grad_shard
            )
            handle.flat_param._saved_grad_shard += new_sharded_grad
        else:
            handle.flat_param._saved_grad_shard = new_sharded_grad

    def backward(self, tensor: torch.Tensor | tuple[tuple[torch.Tensor], torch.Tensor]):
        """
        Stream semantics of backward pass:
        https://pytorch.org/docs/stable/notes/cuda.html
        """

        self.unshard(HandleTrainingState.BACKWARD_PRE)
        self._param_handle.prepare_gradient_for_backward()
        # wait for unshard event to complete
        torch.cuda.current_stream().wait_event(self._unshard_param_event)

        # if there is a previous layer, prefetch unshard it.
        if self._prev_layer is not None:
            self._prev_layer.unshard(HandleTrainingState.BACKWARD_PRE)

        if isinstance(tensor, torch.Tensor):
            tensor.backward()
        else:
            output, gradients = tensor
            torch.autograd.backward(output, gradients)
        self._backward_event.record(torch.cuda.current_stream())

        self.reshard()
