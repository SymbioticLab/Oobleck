import torch
import torch.fx

from typing import Type, Tuple, List

from deepspeed import comm as dist
from deepspeed.runtime.engine import (
    MEMORY_OPT_ALLREDUCE_SIZE,
    split_half_float_double_sparse,
)
from deepspeed.ops.op_builder.utils import UtilsBuilder

from torch.utils.checkpoint import checkpoint as checkpoint_fn
from torch.distributed import ProcessGroup

from transformers import TrainingArguments


def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
    if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
        return False
    if next(layer.parameters(), None) is None:
        return False
    return True


class Layer(torch.nn.Module):
    def __init__(self, index: int, layer: torch.fx.GraphModule, training_args: TrainingArguments):
        super().__init__()
        self.index = index
        self.add_module("layer", layer)
        self.checkpointable = False

        # TODO: will be used for fp16/bf16
        self.training_args = training_args

        # Load pre-installed or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten

    def set_checkpointable(self, checkpointable: bool):
        self.checkpointable = checkpointable

    def forward(self, *args):
        if self.checkpointable:
            return checkpoint_fn(self.layer, *args)
        else:
            return self.layer(*args)

    def _do_allreduce(
        self, bucket: List[torch.Tensor], process_group: Type[ProcessGroup]
    ):
        tensor: torch.Tensor = self.flatten(bucket)
        tensor.mul_(1.0 / dist.get_world_size(group=process_group))
        dist.all_reduce(tensor, group=process_group)

        for buf, synced in zip(bucket, self.unflatten(tensor, bucket)):
            buf.copy_(synced)

    def reduce_gradients(self, process_group: Type[ProcessGroup]):
        """
        Reduce gradients of this layer with processes in the process_group.

        Args:
            process_group (Type[ProcessGroup]): Process group that
                has all processes having this layer.
        """

        elements_per_buffer = MEMORY_OPT_ALLREDUCE_SIZE
        gradients = [
            param.grad.data
            for _, param in self.layer.named_parameters()
            if param.grad is not None
        ]

        split_buckets: List[
            Tuple[torch.dtype, List[torch.Tensor]]
        ] = split_half_float_double_sparse(gradients)
        for bucket_tuple in split_buckets:
            _bucket_type, bucket = bucket_tuple

            small_bucket = []
            numel = 0
            for tensor in bucket:
                small_bucket.append(tensor)
                numel += tensor.numel()
                if numel > elements_per_buffer:
                    self._do_allreduce(small_bucket, process_group)
                    small_bucket = []
                    numel = 0

            # Finish reduce operation if small bucket remains.
            if len(small_bucket) > 0:
                self._do_allreduce(small_bucket, process_group)
