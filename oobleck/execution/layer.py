import torch
import torch.distributed
import torch.fx
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy


def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
    if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
        return False
    if any(isinstance(m, torch.nn.CrossEntropyLoss) for _, m in layer.named_modules()):
        return False
    if next(layer.parameters(), None) is None:
        return False
    return True


class Layer(torch.nn.Module):
    def __init__(
        self,
        layer: torch.fx.GraphModule,
        process_group: torch.distributed.ProcessGroup,
    ):
        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        layer.to(device)
        if is_checkpointable(layer):
            layer = checkpoint_wrapper(layer)

        self._param_handle = FlatParamHandle(
            params=layer.parameters(),
            fully_sharded_module=layer,
            device=device,
            sharding_strategy=HandleShardingStrategy.NO_SHARD,
            offload_params=False,
            mp_param_dtype=torch.float32,  # TODO: change to bf16
            mp_reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
            process_group=process_group,
            use_orig_params=False,
        )
        self._param_handle.shard()
        self._param_handle.init_flat_param_attributes()

    def forward(self, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        return self._param_handle._fully_sharded_module(*input)

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
