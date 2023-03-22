import os
import math
import time
import functools
import torch
import torch.distributed as dist

from ast import literal_eval
from pathlib import Path
from deepspeed.utils.logging import logger
from typing import List, Dict, Any, Tuple, Optional

from oobleck.module.model import OobleckModel
from oobleck.module.layer import Layer


PROFILE_CACHE = "/tmp/oobleck/profiles"
num_warmup = 2
num_iteration = 3


class LayerExecutionResult:
    def __init__(
        self,
        layer_index: int,
        forward: float,
        backward: float,
        allreduce_in_node: Dict[int, float],
        allreduce_cross_nodes: Dict[int, float],
        num_elements: int,
    ):
        self.index = layer_index
        self.forward = forward
        self.backward = backward
        self.allreduce_in_node = allreduce_in_node
        self.allreduce_cross_nodes = allreduce_cross_nodes
        self.num_elements = num_elements


def return_cache_if_exist(profile_type: str):
    def get_cache_if_exists(cache_path: str) -> Optional[dict]:
        file = Path(cache_path)
        if file.is_file():
            logger.info("Loading cache %s", cache_path)
            with file.open(mode="r") as f:
                return literal_eval(f.read())

        return None

    def store_cache(cache_path: str, object: Any):
        if dist.is_initialized() and int(os.environ["LOCAL_RANK"]) != 0:
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
            cache_path = f"{PROFILE_CACHE}/{s.model.model_name}/{profile_type}"
            cache = get_cache_if_exists(cache_path)
            if cache:
                return cache
            result = func(s, *args, **kwargs)
            store_cache(cache_path, result)
            return result

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
        """Profile the given model and return a list of execution result
        per layer.
        ExecutionResult includes forward/backward latency, allreduce in node,
        and allreduce across nodes.

        Returns:
            List[LayerExecutionResult]: A list of execution results per layer.
        """
        dist_initialized = dist.is_initialized()
        if not dist_initialized:
            dist.init_process_group(backend="nccl")

        # forward/backward execution
        layer_execution_result = self._profile_execution_layers()
        allreduce_across_nodes = self._profile_allreduce_across_nodes()
        allreduce_in_node = self._profile_allreduce_in_node()

        results: List[LayerExecutionResult] = []
        for layer, execution, ar_in_node, ar_across_nodes in zip(
            self.model.model,
            layer_execution_result,
            allreduce_in_node,
            allreduce_across_nodes,
        ):
            results.append(
                LayerExecutionResult(
                    layer.index,
                    execution["forward"],
                    execution["backward"],
                    ar_in_node,
                    ar_across_nodes,
                    execution["num_elements"],
                )
            )

        dist.barrier()
        if not dist_initialized:
            dist.destroy_process_group()
        return results

    @return_cache_if_exist("layers")
    def _profile_execution_layers(self) -> List[Dict[str, float]]:
        assert dist.is_initialized()

        results: List[List[int]] = [[0, 0, 0]] * len(self.model.model)
        if dist.get_rank() == 0:
            for i in range(num_warmup + num_iteration):
                logger.info(f"Profiling layer execution ltency: {i} iteration")
                input = tuple(self.model.sample_inputs.values())

                for idx, layer in enumerate(self.model.model):
                    if isinstance(input, tuple):
                        input = tuple(
                            [
                                t.detach().to("cuda")
                                if isinstance(t, torch.Tensor)
                                else t
                                for t in input
                            ]
                        )
                    else:
                        input = input.detach().to("cuda")

                    gpu_layer = layer.to("cuda")
                    torch.cuda.synchronize()

                    start = time.time()
                    output = gpu_layer(*input)
                    torch.cuda.synchronize()
                    end = time.time()
                    input = output
                    forward = end - start

                    del gpu_layer
                    if i < num_warmup:
                        continue

                    results[idx][0] += forward * 1000
                    results[idx][1] += forward * 2000
                    if results[idx][2] == 0:
                        results[idx][2] = sum(
                            [p.numel() for p in layer.parameters() if p.requires_grad]
                        )

        for result in results:
            result[0] /= num_iteration
            result[1] /= num_iteration

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {"forward": result[0], "backward": result[1], "num_elements": result[2]}
            for result in results.tolist()
        ]

    @staticmethod
    def profile_allreduce_layer(
        layer: Layer, process_group: dist.ProcessGroup
    ) -> float:
        numel = sum([p.numel() for p in layer.parameters()])
        tensor = torch.zeros(numel, dtype=torch.float32, device="cuda")

        dist.barrier(process_group)
        start = time.time()
        dist.all_reduce(tensor, group=process_group)
        dist.barrier(process_group)
        end = time.time()

        del tensor
        return (end - start) * 1000

    @return_cache_if_exist("allreduce_cross_nodes")
    def _profile_allreduce_across_nodes(self) -> List[Dict[int, float]]:
        """Profile allreduce latency across nodes,
        \# nodes = 2, 3, ... N.
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            List[Dict[int, float]]: A list of allreduce latency,
            where key is the number of nodes and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(
            f"Profile allreduce acorss {os.environ['WORLD_SIZE']} nodes latency"
        )

        num_gpus_per_node = torch.cuda.device_count()
        ranks = list(range(0, dist.get_world_size(), num_gpus_per_node))

        process_groups: List[Tuple(bool, dist.ProcessGroup)] = []
        for i in range(1, len(ranks) + 1):
            pg_ranks = ranks[:i]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: List[List[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.model))
        ]
        for layer_index, layer in enumerate(self.model.model):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {len(ranks[:i]) + 1: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]

    @return_cache_if_exist("allreduce_in_node")
    def _profile_allreduce_in_node(self) -> List[Dict[int, float]]:
        """Profile allreduce latency between GPUs in node,
        \# nodes = 1, 2, 4, ....
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            List[Dict[int, float]]: A list of allreduce latency,
            where key is the number of GPUs and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(f"Profile allreduce within a node latency")

        num_gpus_per_node = torch.cuda.device_count()
        # 1, 2, 4, 8, ...
        num_gpus_list = [2**i for i in range(int(math.log2(num_gpus_per_node)) + 1)]
        ranks = list(range(num_gpus_per_node))

        process_groups: List[Tuple(bool, dist.ProcessGroup)] = []
        for i in range(len(num_gpus_list)):
            pg_ranks = ranks[: num_gpus_list[i]]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: List[List[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.model))
        ]
        for layer_index, layer in enumerate(self.model.model):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {num_gpus_list[i]: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]
