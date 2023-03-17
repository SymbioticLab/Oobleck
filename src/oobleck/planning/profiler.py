import os
import math
import time
import functools
import torch
import torch.distributed as dist

from ast import literal_eval
from pathlib import Path
from deepspeed.utils.logging import logger
from typing import List, Dict, Any, Optional

from oobleck.module.model import OobleckModel
from oobleck.module.layer import Layer


PROFILE_CACHE = "/tmp/oobleck/profiles"


class LayerExecutionResult:
    def __init__(
        self,
        layer_index: int,
        forward: float,
        backward: float,
        allreduce_in_node: float,
        allreduce_cross_nodes: float,
        num_elements: int,
    ):
        self.index = layer_index
        self.forward = forward
        self.backward = backward
        self.allreduce_in_node = allreduce_in_node
        self.allreduce_cross_nodes = allreduce_cross_nodes
        self.num_elements = num_elements


num_warmup = 2
num_iteration = 3


def return_cache_if_exist(profile_type: str):
    def get_cache_if_exists(cache_path: str) -> Optional[dict]:
        file = Path(cache_path)
        if file.is_file():
            logger.info("Loading cache %s", cache_path)
            with file.open(mode="r") as f:
                return literal_eval(f.read())

        return None

    def store_cache(cache_path: str, object: Any):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        path = Path(cache_path)
        if not path.parent.exists():
            os.makedirs(path.parent, exist_ok=True)

        logger.info("Storing cache %s", cache_path)

        with path.open(mode="w") as f:
            f.write(str(object))
            f.flush()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(s, *args, **kwargs):
            cache_path = f"{PROFILE_CACHE}/{s.model_name}/{profile_type}"
            cache = get_cache_if_exists(cache_path)
            if cache:
                return cache
            result = func(s, *args, **kwargs)
            store_cache(result, cache_path)
            return cache

        return wrapper

    return decorator


class Profiler:
    """Oobleck Profiler that profiles execution latency, allreduce latency in node,
    allreduce latency across node for each layer of the model.

    To support large model profiling, we offload parameters layer by layer.
    """

    def __init__(self, model: OobleckModel):
        self.model = model

    def profile(self) -> List[LayerExecutionResult]:
        if dist.is_initialized():
            dist.destroy_process_group()
        layer_execution_result = self._profile_execution_layers()

        dist.init_process_group()
        num_gpus_per_node = torch.cuda.device_count()
        for start_index in range(0, dist.get_world_size(), num_gpus_per_node):
            ranks = range(start_index, start_index + num_gpus_per_node)
            process_group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                my_local_process_group = process_group
        allreduce_in_node = self._profile_allreduce_in_node(my_local_process_group)

        # TODO: get allreduce cross nodes
        world_ranks = list(range(dist.get_world_size()))
        for i in range(num_gpus_per_node):
            ranks = world_ranks[i::num_gpus_per_node]
            process_group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                my_cross_process_group = process_group
        allreduce_cross_node = self._profile_allreduce_across_nodes(
            my_cross_process_group
        )

        results: List[LayerExecutionResult] = []
        for layer, execution, ar_in_node, ar_cross_node in zip(
            self.model.model,
            layer_execution_result,
            allreduce_in_node,
            allreduce_cross_node,
        ):
            results.append(
                LayerExecutionResult(
                    layer.index,
                    execution["forward"],
                    execution["backward"],
                    ar_in_node,
                    ar_cross_node,
                )
            )
        dist.destroy_process_group()
        return results

    @return_cache_if_exist("layers")
    def _profile_execution_layers(self) -> List[Dict[str, float]]:
        table = [{"forward": 0, "backward": 0} for _ in range(len(self.model.model))]

        for i in range(num_warmup + num_iteration):
            logger.info(f"Profiling layer executio ltency: {i} iteration")
            input = tuple(self.model.dummy_inputs.values())

            for idx, layer in enumerate(self.model.model):
                if isinstance(input, tuple):
                    input = tuple(
                        [
                            t.detach().to("cuda") if isinstance(t, torch.Tensor) else t
                            for t in input
                        ]
                    )
                else:
                    input = input.detach().to("cuda")

                start = time.time()
                output = layer(*input)
                torch.cuda.synchronize()
                end = time.time()
                input = output

                forward = end - start

                if i < num_warmup:
                    continue

                table[idx]["forward"] += forward * 1000
                table[idx]["backward"] += forward * 2000
                if "numel" not in table[idx]:
                    table[idx]["numel"] = sum(
                        [p.numel() for p in layer.parameters() if p.requires_grad]
                    )

        return table

    @staticmethod
    def profile_allreduce_layer(
        layer: Layer, process_group: dist.ProcessGroup
    ) -> float:
        numel = sum([p.numel() for p in layer.parameters()])
        tensor = torch.zeros(numel, dtype=torch.float32, device="cuda")

        dist.barrier()
        start = time.time()
        dist.all_reduce(tensor, group=process_group)
        end = time.time()

        del tensor
        return (end - start) * 1000

    @return_cache_if_exist("allreduce_corss_nodes")
    def _profile_allreduce_across_nodes(
        self, ranks: List[int]
    ) -> List[Dict[int, float]]:
        assert dist.is_initialized()
        logger.info(f"Profile allreduce acorss nodes latency")

        results: List[Dict[int, List[float]]] = []
        for layer in self.model.model:
            result: Dict[int, List[float]] = {}
            for num_nodes in range(1, len(ranks)):
                ranks_pg = ranks[:num_nodes]
                process_group = dist.new_group(ranks_pg)
                result[num_nodes] = Profiler.profile_allreduce_layer(
                    layer, process_group
                )
                dist.destroy_process_group(process_group)
            results.append(result)

        return results

    @return_cache_if_exist("allreduce_in_node")
    def _profile_allreduce_in_node(
        self, ranks: List[int], num_gpus_per_node: int
    ) -> List[Dict[int, float]]:
        assert dist.is_initialized()
        logger.info(f"Profile allreduce within a node latency")

        results: List[Dict[int, List[float]]] = []
        for layer in self.model.model:
            result: Dict[int, List[float]] = {}
            num_gpuss = [2**i for i in range(int(math.log2(num_gpus_per_node)))]
            for num_gpus in num_gpuss:
                ranks_pg = ranks[:num_gpus]
                process_group = dist.new_group(ranks_pg)
                result[num_gpus] = Profiler.profile_allreduce_layer(
                    layer, process_group
                )
                dist.destroy_process_group(process_group)
            results.append(result)

        return results
