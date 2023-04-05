import math
import torch
import time

from collections import defaultdict
from typing import List, Tuple, Dict
from deepspeed.utils.logging import logger

from oobleck.planning.profiler import LayerExecutionResult, get_profile_results
from oobleck.module.model import OobleckModel
from oobleck.utils.singleton import Singleton


class StageExecutionResult:
    """Execution result of one stage on the given device mesh."""

    def __init__(self, layer_exec: List[LayerExecutionResult], device_num: int):
        """
        Args:
            layer_exec (List[LayerExecutionResult]): Each layer execution
            device_num (int): _description_
        """
        assert device_num <= torch.cuda.device_count(), (
            "Assuming this node is used for training as well, "
            f"number of GPUs {torch.cuda.device_count()} "
            f"is not enough to measure communication with {device_num} GPUs."
        )

        self.layer_indicies = (layer_exec[0].index, layer_exec[-1].index + 1)
        self.layer_exec = layer_exec
        self.device_num = device_num

        self.forward = 0
        self.backward = 0
        self.allreduce_across_nodes = defaultdict(float)
        self.num_elements = 0

        for l in layer_exec:
            self.forward += l.forward / self.device_num
            self.backward += l.backward / self.device_num

            if self.device_num > 1:
                self.forward += l.allreduce_in_node[self.device_num]
                self.backward += l.allreduce_in_node[self.device_num]

            self.num_elements += l.num_elements

            for num_replica, ar in l.allreduce_cross_nodes.items():
                self.allreduce_across_nodes[num_replica] += ar

    @property
    def memory_consumption(self) -> int:
        # TODO: consider activation as well.
        return (
            self.num_elements
            * 12
            * (torch.finfo(torch.float32).bits // 8)
            / self.device_num
        )

    def __repr__(self) -> str:
        return (
            f"(StageExecution [{self.layer_indicies[0]}, {self.layer_indicies[1]}) "
            f"with {self.device_num} GPUs)"
        )


class DCExecutionResult:
    """A class wrapper of a list of the stages.
    It splits the problem into two subproblems
    and uses divide and conquer to get the result.
    """

    def __init__(self, stages: List[StageExecutionResult] = []):
        self.stages = stages

        # if len(stages) == 0 or any(
        #     stage.memory_consumption / stage.device_num
        #     > torch.cuda.get_device_properties("cuda:0").total_memory
        #     for stage in stages
        # ):
        if len(stages) == 0:
            self.e1 = math.inf
            self.e2 = math.inf
            self.kstar = -1
            self.e3 = math.inf
            return

        self.e1 = sum([stage.forward + stage.backward for stage in stages])
        self.kstar = DCExecutionResult.get_kstar(self.stages)
        app = 4 * len(self.stages)
        self.e2 = (app - len(self.stages) + self.kstar - 1) * (
            stages[self.kstar].forward + stages[self.kstar].backward
        )
        self.e3 = sum(
            [stage.forward + stage.backward for stage in stages[self.kstar :]]
        )

    # argmax without numpy: https://github.com/cjohnson318/til/blob/main/python/argmax-without-numpy.md
    @staticmethod
    def get_kstar(stages: List[StageExecutionResult]) -> int:
        return max(
            range(len(stages)), key=lambda i: stages[i].forward + stages[i].backward
        )

    def get_e(self) -> float:
        return self.e1 + self.e2 + self.e3

    @classmethod
    def create_from_stage(cls, stage: StageExecutionResult) -> "DCExecutionResult":
        """A function for conquer phase."""
        return cls([stage])

    @classmethod
    def merge_results(
        cls, e_left: "DCExecutionResult", e_right: "DCExecutionResult"
    ) -> "DCExecutionResult":
        """A function for Combination phase."""
        stages = e_left.stages + e_right.stages
        result = cls(stages)

        # Verification of DC combine equations
        assert result.e1 == e_left.e1 + e_right.e1, "e1 calculation fail."
        assert result.kstar in [
            e_left.kstar,
            e_right.kstar + len(e_left.stages),
        ], "kstar calculation fail."
        if result.kstar == e_left.kstar:
            stage_kstar = e_left.stages[e_left.kstar]
            added_app = 4 * len(e_right.stages)
            assert result.e2 == e_left.e2 + (
                added_app - len(e_right.stages) + (result.kstar - e_left.kstar)
            ) * (stage_kstar.forward + stage_kstar.backward), "e2 calculation fail."
            assert result.e3 == e_left.e3 + e_right.e1, "e3 calculation fail."
        else:
            stage_kstar = e_right.stages[e_right.kstar]
            added_app = 4 * len(e_left.stages)
            assert result.e2 == e_right.e2 + (
                added_app - len(e_left.stages) + (result.kstar - e_right.kstar)
            ) * (stage_kstar.forward + stage_kstar.backward), "e2 calculation fail."
            assert result.e3 == e_right.e3, "e3 calculation fail."
        return result


@Singleton
class Planner:
    """Oobleck paper section 4.1.2. Divide and Conquer Algorithm implementation
    For each PipelineSpec, planner generates
    """

    def __init__(self, model: OobleckModel):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

        self.model_layers = get_profile_results(model)

    def get_cache_hit_ratio(self, clear: bool = False) -> float:
        hit_ratio = 0.0
        if not self.hit_count + self.miss_count == 0:
            hit_ratio = (self.hit_count / (self.hit_count + self.miss_count)) * 100
        if clear:
            self.hit_count = 0
            self.miss_count = 0
        return hit_ratio

    def get_execution_plan(
        self, num_nodes: int, num_gpus_per_node: int
    ) -> DCExecutionResult:
        max_num_stages = len(self.model_layers)
        min_num_stages = num_nodes

        start = time.time()
        e = DCExecutionResult()
        for s in range(min_num_stages, max_num_stages + 1):
            # Iterate number of stages between # nodes (minimum)
            # and # layers (maximum) and get the best execution plan.
            new_e = self._run_divide_and_conquer(
                s, (0, len(self.model_layers)), num_nodes, num_gpus_per_node
            )

            if new_e.get_e() < e.get_e():
                e = new_e

        end = time.time()

        logger.info(
            "Getting execution plan with %d nodes in %.2f ms. Cache hit ratio: %.2f%%",
            num_nodes,
            (end - start) * 1000,
            self.get_cache_hit_ratio(clear=True),
        )

        return e

    def _run_divide_and_conquer(
        self,
        num_stages: int,
        layer_indices: Tuple[int, int],
        num_nodes: int,
        num_gpus_per_node: int,
    ) -> DCExecutionResult:
        """Get execution plan for this subproblem using divide and conquer."""
        # return cached result if it exists
        key = (num_stages, layer_indices, (num_nodes, num_gpus_per_node))
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]

        layers = self.model_layers[layer_indices[0] : layer_indices[1]]

        self.miss_count += 1
        e = DCExecutionResult()

        # ============================================
        # filter infeasible case
        # ============================================
        if num_stages > (layer_indices[1] - layer_indices[0]):
            # if number of stages is less than number of layers
            self.cache[key] = e
            return e

        if num_nodes == 1:
            if num_gpus_per_node < num_stages:
                # GPU cannot be shared by several stages
                self.cache[key] = e
                return e

            if num_gpus_per_node < num_stages:
                # At least one GPU should be assigned to each stage
                self.cache[key] = e
                return e

            if num_stages == 1 and not math.log2(num_gpus_per_node).is_integer():
                # One stage cannot have non-power-of-two GPUs
                self.cache[key] = e
                return e

        elif num_nodes > num_stages:
            # Two or more node cannot be assigned to the same stage
            self.cache[key] = e
            return e

        # ============================================
        # base case
        # ============================================
        if num_stages == 1:
            assert num_nodes == 1
            stage = StageExecutionResult(layers, num_gpus_per_node)
            e = DCExecutionResult.create_from_stage(stage)
            self.cache[key] = e
            return e

        # ============================================
        # divide and combine
        # ============================================
        for k in range(1, len(layers)):
            if num_nodes == 1:
                # Split GPUs in a node.
                for num_gpus_left in range(1, num_gpus_per_node):
                    for num_stages_left in range(1, num_stages):
                        e_left = self._run_divide_and_conquer(
                            num_stages=num_stages_left,
                            layer_indices=(layer_indices[0], k),
                            num_nodes=1,
                            num_gpus_per_node=num_gpus_left,
                        )
                        e_right = self._run_divide_and_conquer(
                            num_stages=num_stages - num_stages_left,
                            layer_indices=(k, layer_indices[1]),
                            num_nodes=1,
                            num_gpus_per_node=num_gpus_per_node - num_stages_left,
                        )
                        if e_left.get_e() == math.inf or e_right.get_e() == math.inf:
                            continue

                        new_e = DCExecutionResult.merge_results(e_left, e_right)
                        if new_e.get_e() < e.get_e():
                            e = new_e
            else:
                for num_nodes_left in range(1, num_nodes):
                    for num_stages_left in range(1, num_stages):
                        e_left = self._run_divide_and_conquer(
                            num_stages=num_stages_left,
                            layer_indices=(layer_indices[0], k),
                            num_nodes=num_nodes_left,
                            num_gpus_per_node=num_gpus_per_node,
                        )
                        e_right = self._run_divide_and_conquer(
                            num_stages=num_stages - num_stages_left,
                            layer_indices=(k, layer_indices[1]),
                            num_nodes=num_nodes - num_nodes_left,
                            num_gpus_per_node=num_gpus_per_node,
                        )
                        if e_left.get_e() == math.inf or e_right.get_e() == math.inf:
                            continue

                        new_e = DCExecutionResult.merge_results(e_left, e_right)
                        if new_e.get_e() < e.get_e():
                            e = new_e

        self.cache[key] = e
        return e


class PipelineSpec:
    """
    A specification of the pipeline representation.
    Oobleck represents the total device cluster as a linear combination
    of several distinct heterogeneous :class:`PipelineSpec`s to cover all availale GPUs.

    Based on the given fault tolerance spec and maximum available number of nodes,
    several PipelineSpecs are created in advance and the planner analyzes the optimal
    execution plan for each PipelineSpec under the given number of GPUs.

    Exploiting the Frobenius problem, it is guaranteed to represent any feasible number of nodes
    as a linear of combination of PipelineSpecs with consecutive number of nodes.
    """

    def __init__(self, num_nodes: int, num_gpus_per_node: int, model: OobleckModel):
        assert (
            num_nodes > 0 and num_gpus_per_node > 0
        ), f"Number of nodes or GPUs cannot be 0 or negative - # nodes: {num_nodes}, # GPUs: {num_gpus_per_node}"

        # TODO: currently does not consider multi GPU per node.
        assert num_gpus_per_node == 1

        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.model = model

        self.planner = Planner(self.model)
        self.optimal_plan = self.planner.get_execution_plan(
            self.num_nodes, self.num_gpus_per_node
        )

        # Translate `DCExecutionResult` into layer-rank map
        # layer_spec has a list of ranks, each of which is mapped to each layer.
        layer_spec: List[int] = [None] * len(model.model)
        num_used_gpus = 0
        for stage in sorted(
            self.optimal_plan.stages, key=lambda s: s.layer_indicies[0]
        ):
            for layer_index in range(stage.layer_indicies[0], stage.layer_indicies[1]):
                gpus = list(range(num_used_gpus, num_used_gpus + stage.device_num))
                assert (
                    len(gpus) == 1
                ), "TODO: Currently only one GPU can be assigned to each layer."
                layer_spec[layer_index] = gpus[0]
            num_used_gpus += stage.device_num

        assert all(
            spec is not None for spec in layer_spec
        ), f"Some layer has no plan: {layer_spec}"
        self.layer_spec = layer_spec

    # Use PipelineSpec as a key of dictionary in dynamic programming
    def __hash__(self) -> int:
        return hash(tuple(self.layer_spec))

    def __eq__(self, other: "PipelineSpec") -> bool:
        if (
            len(self.layer_spec) != len(other.layer_spec)
            or self.num_nodes != other.num_nodes
            or self.num_gpus_per_node != other.num_gpus_per_node
        ):
            return False

        return all(
            my_l == his_l for my_l, his_l in zip(self.layer_spec, other.layer_spec)
        )

    def __repr__(self) -> str:
        return f"(PipelineSpec: {self.num_nodes} nodes)"

    @classmethod
    def create(
        cls,
        ft_spec: int,
        max_num_nodes: int,
        num_gpus_per_node: int,
        model: OobleckModel,
    ) -> List["PipelineSpec"]:
        """Oobleck paper section 4.1. Configuring PipelineSpecs implementation
        Generates the list of :class:`oobleck.planning.pipeline_spec.PipelineSpec`s
        with heterogeneous number of nodes specifications, a linear combination of
        which can represent any number N (min_num_nodes <= N <= max_num_nodes).

        Oobleck exploits the Frobenius problem and creates a list of `PipelineSpec`s
        based on the constraints:
        1. \# of `PipelineSpec`s > n0 - 1
        2. ni's are consecutive integers (ni + 1 = n(i+1))
        We define n0 = minimum number of nodes ths is required to hold states for training.

        Args:
            ft_spec (int): Fault tolerant spec.
                Oobleck tries to create at least ft_spec + 1 model replica.
            max_num_nodes (int): Maximum # nodes in the cluster.

        Returns:
            List[PipelineSpec]: List of `PipelineSpec`s.
                Length of the list is the number of `PipelineSpec`s, p.
        """
        assert ft_spec >= 0, "Fault tolerance spec must not be negative."

        # TODO: currently required memory calculation is not correct.
        required_memory = model.total_num_params * 12 * 6
        gpu_memory = torch.cuda.get_device_properties("cuda:0").total_memory
        required_min_gpus = math.ceil(required_memory / gpu_memory)
        logger.info(
            f"Required memory: {required_memory/1e6:.2f} MB, Capacity: {gpu_memory/1e6:.2f} MB"
        )
        min_pipeline_spec = math.ceil(required_min_gpus / num_gpus_per_node)
        min_pipeline_spec = max(min_pipeline_spec, 1)
        assert (
            ft_spec + 1
        ) * min_pipeline_spec <= max_num_nodes, f"Maximum # nodes ({max_num_nodes}) cannot be smaller than minimum # nodes ({min_pipeline_spec})."
        if (ft_spec + 1) * min_pipeline_spec > max_num_nodes:
            logger.warning(
                "The number of nodes is not enough to provide at least ft_spec + 1 copy of the model."
                "Oobleck may fail to provide fault tolerancy if continue."
            )

        # p = n0 - 1
        # num_pipeline_specs = min_pipeline_spec
        # p: length between n0 and N - fn0
        max_pipeline_spec = max_num_nodes - ft_spec * min_pipeline_spec
        if max_pipeline_spec < 2:
            max_pipeline_spec = 2

        pipeline_specs = list(range(min_pipeline_spec, max_pipeline_spec + 1))
        assert all(
            num_nodes <= max_num_nodes for num_nodes in pipeline_specs
        ), "Some PipelineSpec needs to have more # nodes than maximum # nodes (impossible)."

        results = []
        for num_nodes in pipeline_specs:
            try:
                spec = cls(num_nodes, num_gpus_per_node, model)
                results.append(spec)
            except AssertionError:
                pass
        return results

        return [
            cls(num_nodes, num_gpus_per_node, model) for num_nodes in pipeline_specs
        ]
