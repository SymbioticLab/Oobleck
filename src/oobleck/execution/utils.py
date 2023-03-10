import torch
from typing import Union, Iterable


def zero_grads(inputs: Union[torch.Tensor, Iterable[torch.Tensor]]):
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.data.zero_()
    else:
        for t in inputs:
            if t.grad is not None:
                t.grad.data.zero_()
