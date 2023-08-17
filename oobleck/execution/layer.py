from __future__ import annotations

import copy

import torch
import torch.distributed
import torch.fx
from accelerate.utils.modeling import set_module_tensor_to_device
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy


def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
    if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
        return False
    if any(isinstance(m, torch.nn.CrossEntropyLoss) for _, m in layer.named_modules()):
        return False
    if next(layer.parameters(), None) is None:
        return False
    return True


def init_tensors(layer: torch.fx.GraphModule, device: torch.device):
    """
    Initialize meta tensors and move it to GPU.
    TODO: must use checkpointed data
    """
    for param_name, param in layer.named_parameters():
        set_module_tensor_to_device(layer, param_name, device, torch.rand(param.shape))

    for buffer_name, buffer in layer.named_buffers():
        set_module_tensor_to_device(
            layer, buffer_name, device, torch.rand(buffer.shape)
        )


class Layer(torch.nn.Module):
    @classmethod
    def create_layer_from_layer(
        cls,
        existing_layer: Layer,
        process_group: torch.distributed.ProcessGroup,
    ) -> Layer:
        assert torch.distributed.get_rank(process_group) >= 0

        layer = cls.__new__(cls)
        layer.layer_id = existing_layer.layer_id
        layer._rank_index = torch.distributed.get_rank(process_group)
        layer._group_size = torch.distributed.get_world_size(process_group)
        layer._param_handle = existing_layer._param_handle
        layer._param_handle.process_group = process_group
        layer.pre_stream = existing_layer.pre_stream
        layer.post_stream = existing_layer.post_stream

        layer.register_forward_pre_hook(layer.pre_forward_hook)
        layer.register_forward_hook(layer.post_forward_hook)
        layer.register_full_backward_pre_hook(layer.pre_backward_hook)

        return layer

    def remove_tensors(self):
        if self._param_handle.flat_param.grad is not None:
            self._param_handle.flat_param.grad.data = torch.tensor([])
        self._param_handle.flat_param.data = torch.tensor([])

    def __init__(
        self,
        layer_id: int,
        layer: torch.fx.GraphModule,
        process_group: torch.distributed.ProcessGroup,
        pre_stream: torch.cuda.Stream,
        post_stream: torch.cuda.Stream,
    ):
        super().__init__()

        assert torch.distributed.get_rank(process_group) >= 0

        device = torch.device("cuda", torch.cuda.current_device())
        self.layer_id = layer_id
        self._rank_index = torch.distributed.get_rank(process_group)
        self._group_size = torch.distributed.get_world_size(process_group)

        self.pre_stream = pre_stream
        self.post_stream = post_stream

        layer = copy.deepcopy(layer)
        init_tensors(layer, device)
        if is_checkpointable(layer):
            layer = checkpoint_wrapper(layer)

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
        self._param_handle.init_flat_param_attributes()

        self.register_forward_pre_hook(self.pre_forward_hook)
        self.register_forward_hook(self.post_forward_hook)
        self.register_full_backward_pre_hook(self.pre_backward_hook)

    def unshard_params(self, state: HandleTrainingState):
        assert state in [
            HandleTrainingState.FORWARD,
            HandleTrainingState.BACKWARD_PRE,
        ], f"Invalid training state {self._param_handle._training_state}"
        self._param_handle._training_state = state

        if self._param_handle._sharding_strategy == HandleShardingStrategy.NO_SHARD:
            return

        # all further kernel execution will wait `all_gather_into_tensor()` to finish
        with torch.cuda.stream(self.pre_stream):
            self._param_handle.pre_unshard()
            self._param_handle.unshard()
            self._param_handle.post_unshard()

    def reshard_params(self):
        if (
            self._param_handle._sharding_strategy == HandleShardingStrategy.NO_SHARD
            or self._param_handle.needs_unshard()
        ):
            return

        with torch.cuda.stream(self.pre_stream):
            self._param_handle.reshard(True)
            self._param_handle.post_reshard()

    def forward(self, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        return self._param_handle._fully_sharded_module(*input)

    def pre_forward_hook(self, *unused):
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.pre_forward_hook"
        ):
            self.unshard_params(HandleTrainingState.FORWARD)
            torch.cuda.current_stream().wait_stream(self.pre_stream)
        self.register_post_backward_hooks()

    def post_forward_hook(self, *unused):
        self.reshard_params()

    def pre_backward_hook(self, *unused):
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.pre_backward_hook"
        ):
            self.unshard_params(HandleTrainingState.BACKWARD_PRE)

            self._param_handle._clear_grads_if_needed()
            self._param_handle.prepare_gradient_for_backward()

    def post_backward_hook(self, *unused):
        """
        Code adopted from torch.distributed.fsdp._runtime_utils.py::_post_backward_hook

        Reduce-scatters the gradient of `self._param_handle.flat_param`.

        Precondition: The `FlatParameter`s `.grad` attribute contains
        the unsharded gradient for the local batch.

        Postcondition:
        - If using `NO_SHARD`, then the `.grad` attribute is unchanged
        as unsharded gradients.
        - If using `FULL_SHARD`, then the `_saved_grad_shard` attribute is the
        reduced sharded gradient (accumulating with any existing gradient).
        """
        self._param_handle._training_state = HandleTrainingState.BACKWARD_POST
        self.post_stream.wait_stream(torch.cuda.current_stream())

        # Code adopted from torch.distributed.fsdp._runtime_utils.py::_post_backward_hook
        with torch.cuda.stream(
            self.post_stream
        ), torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.post_backward_hook"
        ):
            if (
                self._param_handle.flat_param.requires_grad is False
                or self._param_handle.flat_param.grad is None
            ):
                return

            self.reshard_params()

            if self._param_handle.uses_sharded_strategy:
                unsharded_grad = self._param_handle.flat_param.grad
                self._param_handle.flat_param.grad = None
                chunks = list(unsharded_grad.chunk(self._group_size))
                numel_to_pad = (
                    self._group_size * chunks[0].numel() - unsharded_grad.numel()
                )
                padded_unsharded_grad = (
                    torch.nn.functional.pad(unsharded_grad, [0, numel_to_pad])
                    if numel_to_pad > 0
                    else unsharded_grad
                )
                new_sharded_grad = torch.empty_like(chunks[0])  # padded

                torch.distributed.reduce_scatter_tensor(
                    new_sharded_grad,
                    padded_unsharded_grad,
                    group=self._param_handle.process_group,
                )

                # Accumulate gradients
                if hasattr(self._param_handle.flat_param, "_saved_grad_shard"):
                    self._param_handle.flat_param._saved_grad_shard += new_sharded_grad
                else:
                    self._param_handle.flat_param._saved_grad_shard = new_sharded_grad

            self._param_handle.flat_param._post_backward_called = True

    def register_post_backward_hooks(self):
        """Code adopted from torch.distributed.fsdp._runtime_utils.py::_register_post_backward_hooks"""
        flat_param = self._param_handle.flat_param

        if flat_param.requires_grad and not hasattr(
            flat_param, "_post_backward_hook_state"
        ):
            # post backward hook
            flat_param = flat_param.expand_as(flat_param)
            assert flat_param.grad_fn is not None

            acc_grad = flat_param.grad_fn.next_functions[0][0]
            assert acc_grad is not None
            hook_handle = acc_grad.register_hook(self.post_backward_hook)
            flat_param._post_backward_hook_state = (acc_grad, hook_handle)

    def remove_post_backward_hooks(self):
        flat_param = self._param_handle.flat_param
        if hasattr(flat_param, "_post_backward_hook_state"):
            acc_grad, hook_handle = flat_param._post_backward_hook_state
            acc_grad.unregister_hook(hook_handle)
            del flat_param._post_backward_hook_state

    def backward(
        self,
        tensor: torch.Tensor
        | tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> None:
        if isinstance(tensor, torch.Tensor):
            loss = tensor
            loss.backward()
        else:
            output, gradients = tensor
            torch.autograd.backward(output, gradients)

    def reduce_gradients(self, process_group: torch.distributed.ProcessGroup):
        torch.cuda.current_stream().wait_stream(self.post_stream)
        self._param_handle.prepare_gradient_for_optim()

        assert torch.distributed.get_rank(process_group) >= 0
        torch.distributed.all_reduce(
            tensor=self._param_handle.flat_param.grad, group=process_group
        )
