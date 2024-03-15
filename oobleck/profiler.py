import csv
import functools
import importlib
import multiprocessing
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from colossalai.shardformer import ShardConfig, ShardFormer
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.distributed import FileStore
from transformers import PretrainedConfig, PreTrainedModel


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
        model_class_name: str,
        config: PretrainedConfig,
        microbatch_size: int,
        base_dir: Path = Path("/tmp/oobleck"),
    ):
        self.model_class_name = model_class_name
        self.model_config = config
        self.microbatch_size = microbatch_size
        self.profile_path = base_dir / tag / "profile" / f"mb_{microbatch_size}.csv"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

        # config_path = self.profile_path.parent / "config.yaml"
        # if config_path.exists():
        #     assert set(config.to_dict().items()) == set(
        #         yaml.safe_load(config_path.read_text()).items()
        #     ), (
        #         "Model config mismatch. "
        #         "Please delete the profile directory and re-run the script or use another tag."
        #     )

        self.profile_result = self.load_profile()

    def load_profile(self) -> list[LayerExecutionResult] | None:
        """Load the profile."""
        if not self.profile_path.exists():
            return None

        try:
            with self.profile_path.open("r") as f:
                reader = csv.DictReader(f)
                return [
                    LayerExecutionResult(
                        layer_index=int(row["layer_index"]),
                        layer_name=row["layer_name"],
                        forward=float(row["forward"]),
                        backward=float(row["backward"]),
                        mem_required=int(row["mem_required"]),
                    )
                    for row in reader
                ]
        except (csv.Error, KeyError):
            return None

    @property
    def mem_consumption(self) -> int:
        """Get the overall memory consumption."""
        if not self.profile_exists():
            raise ValueError("Profile does not exist.")

        return sum(result.mem_required for result in self.profile_result)

    def profile_exists(self) -> bool:
        """Check if the profile exists."""
        return self.profile_result is not None

    @staticmethod
    def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
        """Get a module by its name."""
        names = name.split(".")
        return reduce(getattr, names, model)

    def profile(
        self,
        local_rank: int,
        tensor_parallel_size: int,
        inputs: dict[str, torch.Tensor],
    ):
        """Profile the model."""
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=ModelProfiler._profile_model,
            args=(
                self.model_class_name,
                self.model_config,
                self.profile_path,
                local_rank,
                tensor_parallel_size,
                inputs,
            ),
            daemon=True,
        )
        process.start()
        process.join()

    @staticmethod
    def _profile_model(
        model_class_name: str,
        model_config: PretrainedConfig,
        profile_path: Path,
        local_rank: int,
        tensor_parallel_size: int,
        inputs: dict[str, torch.Tensor],
        warmup: int = 3,
        repeat: int = 5,
    ):
        """Use filestore to profile the model within a node."""

        module = importlib.import_module("transformers")
        module = getattr(module, model_class_name)
        model: PreTrainedModel = module(model_config)
        layers = PipelineTemplate.get_modules(model)

        store_path = profile_path.parent / "store"
        logger.debug(f"Profiler initiating torch.distributed: {store_path}")

        store = FileStore(str(store_path), tensor_parallel_size)
        dist.init_process_group(
            backend="nccl",
            world_size=tensor_parallel_size,
            rank=local_rank,
            store=store,
        )

        logger.debug(
            f"Sharding model with {dist.get_process_group_ranks(dist.group.WORLD)} ranks"
        )

        if tensor_parallel_size > 1:
            shard_config = ShardConfig(
                tensor_parallel_process_group=dist.group.WORLD,
                pipeline_stage_manager=None,
                enable_tensor_parallelism=True,
                enable_flash_attention=False,
            )

            shardformer = ShardFormer(shard_config)
            model, _ = shardformer.optimize(model)

        # Move inputs to cuda
        for name in inputs.keys():
            inputs[name] = inputs[name].to("cuda")
            inputs[name].requires_grad = inputs[name].is_floating_point()

        events: dict[nn.Module, list[torch.cuda.Event]] = {}
        memory: dict[nn.Module, list[int]] = {}

        logger.debug("Profiler started...")

        init_memory = torch.cuda.memory_allocated()
        for layer_name in layers:
            module = ModelProfiler.get_module_by_name(model, layer_name)

            # forward start, forward end, backward start, backward end
            events[module] = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            memory[module] = [0 for _ in range(4)]

            def forward_pre_hook(module_name, module, inputs):
                module.to("cuda")
                memory[module][0] = torch.cuda.memory_allocated() - init_memory
                events[module][0].record()

            def forward_hook(module_name, module, inputs, outputs):
                memory[module][1] = torch.cuda.memory_allocated() - init_memory
                events[module][1].record()
                if not (
                    model.config.tie_word_embeddings
                    and module == model.get_input_embeddings()
                ):
                    module.to("cpu")

            def backward_pre_hook(module_name, module, grad_output):
                module.to("cuda")
                memory[module][2] = torch.cuda.memory_allocated() - init_memory
                events[module][2].record()

            def backward_hook(module_name, module, grad_input, grad_output):
                memory[module][3] = torch.cuda.memory_allocated() - init_memory
                events[module][3].record()

                if not (
                    model.config.tie_word_embeddings
                    and module == model.get_input_embeddings()
                ):
                    module.to("cpu")

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

        for _ in range(repeat):
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()

        logger.debug("Profiler finished.")

        if dist.get_rank() == 0:
            logger.debug(f"Writing results to {profile_path}")
            with profile_path.open("w") as f:
                writer = csv.DictWriter(
                    f, fieldnames=LayerExecutionResult.__annotations__.keys()
                )
                writer.writeheader()

                for index, layer_name in enumerate(layers):
                    module = ModelProfiler.get_module_by_name(model, layer_name)
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

            store_path.unlink()

            config_path = profile_path.parent / "config.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(model_config.to_dict(), f)

        dist.barrier()
        dist.destroy_process_group()
