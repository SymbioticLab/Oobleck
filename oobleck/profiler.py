import copy
import csv
import functools
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class LayerExecutionResult:
    layer_index: int
    layer_name: str
    forward: float
    backward: float
    mem_required: int


class ModelProfiler:
    """A class for profiling a model.

    Profiling includes:
    - Forward and backward latency (in ms) for each layer
    - Maximum memory consumption (in bytes) for each layer

    Args:
        model (nn.Module): The model to be profiled.
        layers (list[str]): A list of layer names to be profiled.
        model must have modules with the given names.
    """

    def __init__(
        self,
        tag: str,
        model: nn.Module,
        layers: list[str],
        base_dir: Path = Path("/tmp/oobleck"),
    ):
        assert all(self.get_module_by_name(model, name) for name in layers)
        self.model = model
        self.layers = layers
        self.profile_path = base_dir / "profiles" / f"{tag}.csv"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def profile_exists(self) -> bool:
        """Check if the profile exists."""
        return self.profile_path.exists()

    @staticmethod
    def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
        """Get a module by its name."""
        names = name.split(".")
        return reduce(getattr, names, model)

    def profile(
        self,
        inputs: dict[str, torch.Tensor],
        warmup: int = 3,
        repeat: int = 5,
    ):
        """Profile the model.

        Returns:
            dict[str, dict[str, float | int]]: A dictionary of layer names and their
                forward and backward latency and memory consumption.
        """

        events: dict[nn.Module, list[torch.cuda.Event]] = {}
        memory: dict[nn.Module, list[int]] = {}
        for name in inputs.keys():
            inputs[name] = inputs[name].cuda()
            inputs[name].requires_grad = inputs[name].is_floating_point()

        torch.cuda.synchronize()
        model = copy.deepcopy(self.model)
        model.train()

        init_memory = torch.cuda.memory_allocated()

        for layer_name in self.layers:
            module = self.get_module_by_name(model, layer_name)

            # forward start, forward end, backward start, backward end
            events[module] = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            memory[module] = [0 for _ in range(4)]

            def forward_pre_hook(module_name, module, inputs):
                print(f"forward_pre_hook for {module_name}")
                module.cuda()
                memory[module][0] = torch.cuda.memory_allocated() - init_memory
                events[module][0].record()

            def forward_hook(module_name, module, inputs, outputs):
                print(f"forward_hook for {module_name}")
                memory[module][1] = torch.cuda.memory_allocated() - init_memory
                events[module][1].record()
                if not (
                    model.config.tie_word_embeddings
                    and module == model.get_input_embeddings()
                ):
                    module.cpu()

            def backward_pre_hook(module_name, module, grad_output):
                print(f"backward_pre_hook for {module_name}")
                module.cuda()
                memory[module][2] = torch.cuda.memory_allocated() - init_memory
                events[module][2].record()

            def backward_hook(module_name, module, grad_input, grad_output):
                print(f"backward_hook for {module_name}")
                memory[module][3] = torch.cuda.memory_allocated() - init_memory
                events[module][3].record()

                if not (
                    model.config.tie_word_embeddings
                    and module == model.get_input_embeddings()
                ):
                    module.cpu()

            module.register_forward_pre_hook(
                functools.partial(forward_pre_hook, layer_name)
            )
            module.register_forward_hook(functools.partial(forward_hook, layer_name))
            module.register_full_backward_pre_hook(
                functools.partial(backward_pre_hook, layer_name)
            )
            module.register_full_backward_hook(
                functools.partial(backward_hook, layer_name)
            )

        with torch.no_grad():
            for _ in range(warmup):
                model(**inputs)

        for i in range(repeat):
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()

        with self.profile_path.open("w") as f:
            writer = csv.DictWriter(
                f, fieldnames=LayerExecutionResult.__annotations__.keys()
            )
            writer.writeheader()

            for index, layer_name in enumerate(self.layers):
                module = self.get_module_by_name(model, layer_name)
                forward = events[module][0].elapsed_time(events[module][1])
                backward = events[module][2].elapsed_time(events[module][3])
                mem_required = memory[module][3]

                result = LayerExecutionResult(
                    layer_index=index,
                    layer_name=layer_name,
                    forward=forward,
                    backward=backward,
                    mem_required=mem_required,
                )
                writer.writerow(asdict(result))
