import torch
import torch.fx

from torch.utils.checkpoint import checkpoint as checkpoint_fn


def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
    if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
        return False
    if next(layer.parameters(), None) is None:
        return False
    return True


class Layer(torch.nn.Module):
    def __init__(self, layer: torch.fx.GraphModule):
        super().__init__()
        self.add_module("graph", layer)

        self.layer = layer
        self.checkpointable = False

    def set_checkpointable(self, checkpointable: bool):
        self.checkpointable = checkpointable

    def forward(self, *args):
        if self.checkpointable:
            return checkpoint_fn(self.layer, *args)
        else:
            return self.layer(*args)

    def reduce_gradients(self):
        pass
