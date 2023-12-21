import torch
from typing import Union, Iterable, Tuple, List

ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}


# helper decorator for initializaton.
# Source: https://stackoverflow.com/a/4104188
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def zero_grads(inputs: Union[torch.Tensor, Iterable[torch.Tensor]]):
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.data.zero_()
    else:
        for t in inputs:
            if t.grad is not None:
                t.grad.data.zero_()
