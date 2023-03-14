import math

from itertools import chain
from typing import List, Iterator
from deepspeed.utils.logging import logger

from oobleck.module.model import OobleckModel


class LayerExecutionSpec:
    def __init__(self, layer_index: int, ranks: List[int]):
        self.layer_index = layer_index
        self.ranks = ranks


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

        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.model = model

        self.layers_spec = self.create_optimal_plan(model)

    def map_ranks_to_spec(self, ranks: List[List[int]]) -> List[LayerExecutionSpec]:
        """Maps given ranks (list of nodes that potentially have multiple ranks as a list)
            to PipelineSpec.

        Args:
            ranks (List[List[int]]): Actual ranks of GPUs in multiple nodes.
                Length must match with PipelineSpec.num_nodes.

        Returns:
            List[LayerExecutionSpec]: list of layer exeecution specification but with
                actual ranks assigned.
        """
        assert len(ranks) == self.num_nodes, "Number of nodes does not match with spec."
        assert all(
            len(ranks_per_node) == self.num_gpus_per_node for ranks_per_node in ranks
        ), "Number of GPUs in some nodes does not match with spec."

        results = []
        flatten_ranks = list(chain.from_iterable(ranks))
        for layer_spec in self.layers_spec:
            results.append(
                LayerExecutionSpec(
                    layer_spec.layer_index, [flatten_ranks[i] for i in layer_spec.ranks]
                )
            )
        return results

    def create_optimal_plan(self, model: OobleckModel) -> List[LayerExecutionSpec]:
        """Create an optimal execution plan with the given number of GPUs
        using profiled model execution information.

        Current Alpha-level implementation: divide layers individually.
        Number of stages is equal to the number of nodes.

        Args:
            num_nodes (int): Number of nodes
            num_gpus_per_node (int): Number of GPUs per node.
                Oobleck assumes all nodes are identical, having same number of GPUs.
            model (OobleckModel): model to profile.

        Returns:
            List[LayerExecutionSpec]: The list of layer specs that the model is divided to.
                Each `LayerExecutionSpec` has its specification (number of GPUs).
        """
        num_layer_per_node = len(model.model) // self.num_nodes
        # num_layers: number of layers that are assigned to each stage.
        num_layers = [num_layer_per_node] * self.num_nodes
        if num_layer_per_node * self.num_nodes < len(model.model):
            num_layers[-1] += len(model.model) - num_layer_per_node * self.num_nodes

        layer_specs = []
        sum = 0
        for i, (node_id, num_layer) in enumerate(
            zip(range(self.num_nodes), num_layers)
        ):
            end_index = sum + num_layer
            for l in range(sum, end_index):
                layer_specs.append(
                    LayerExecutionSpec(
                        l,
                        list(range(i * node_id, i * node_id + self.num_gpus_per_node)),
                    )
                )

            sum += num_layer

        return layer_specs


class PipelineSpecs:
    """
    A wrapper class of the list of :class:`PipelineSpec`.
    It generates the list of :class:`.PipelineSpec`s that can represent any number N that
    min_num_nodes <= N <= max_num_nodes as a linear combination of them.
    """

    def __init__(
        self,
        ft_spec: int,
        min_num_nodes: int,
        max_num_nodes: int,
        num_gpus_per_node: int,
    ):
        """
        Args:
            ft_spec (int): Fault tolerant spec.
                Oobleck tries to create at least ft_spec + 1 model replica.
            min_num_nodes (int): Mininum # nodes to hold the model states for training.
            max_num_nodes (int): Maximum # nodes in the cluster.
            num_gpus_per_node (int): # GPUs per node.
        """
        self.pipeline_specs = None

        assert min_num_nodes > 0, "Minimum # nodes to hold moded states must be > 0."
        assert (
            max_num_nodes >= min_num_nodes
        ), "Maximum # nodes cannot be smaller than minimum # nodes."

        min_req_nodes = min_num_nodes * (ft_spec + 1)
        if max_num_nodes < min_req_nodes:
            logger.warning(
                "The number of nodes is not enough to provide at least ft_spec + 1 copy of the model."
                "Oobleck may fail to provide fault tolerancy if continue."
            )

        # Oobleck's requirements to solve the Frobenius problem
        # 1. p > n[0] - 2 (thus the minimum p is n[0] - 1)
        # 2. n's are contiguous integers (n[i] + 1 = n[i+1])
        # TODO: verify that it is always better to have smaller number of GPUs per PipelineSpecs.
        # (i.e. why we choose minimum p)
        num_pipeline_specs = min_num_nodes - 1
        if num_pipeline_specs < 1:
            num_pipeline_specs = 1

        pipeline_spec_num_nodes = list(
            range(min_num_nodes, min_num_nodes + num_pipeline_specs)
        )
        # Some PipelineSpec cannot be realized with current max_num_nodes requirement.
        assert not any(
            num_nodes > max_num_nodes for num_nodes in pipeline_spec_num_nodes
        ), "Some PipelineSpec needs to have more # nodes than maximum # nodes (impossible)."

        self.ft_spec = ft_spec
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.pipeline_specs = [
            PipelineSpec(num_nodes, num_gpus_per_node)
            for num_nodes in pipeline_spec_num_nodes
        ]

    def __iter__(self) -> Iterator[PipelineSpec]:
        return iter(self.pipeline_specs)

    def __repr__(self) -> str:
        return (
            f"{len(self.pipeline_specs)} PipelineSpecs with # nodes {[p.num_nodes for p in self.pipeline_specs]} "
            f"(# GPUs per node: {self.num_gpus_per_node})"
        )

    def get_num_pipelinespec(self, num_nodes: int) -> List[int]:
        """Return required number of heterogeneous pipelines that
        a linear of combination of the pipelines fills num_nodes.

        Current Alpha-level implementation: always prefer smaller pipelines.
        TODO: analyze the best optimal combination that has the highest throughput.

        Args:
            num_nodes (int): current number of available nodes after failures.

        Returns:
            List[int]: a list of number representing the exact number of
            corresponding pipelines to be deployed.
        """
        if num_nodes > self.max_num_nodes:
            return []

        result = [0] * len(self.pipeline_specs)

        result[0] = math.floor(num_nodes / self.pipeline_specs[0].num_nodes)
        total_assigned_nodes = result[0] * self.pipeline_specs[0].num_nodes
        assert (
            total_assigned_nodes <= num_nodes
        ), f"total assigned nodes {total_assigned_nodes} is not less than total given nodes {num_nodes}"

        smallest_non_zero_pipeline_index = 0
        while total_assigned_nodes < num_nodes:
            while (
                smallest_non_zero_pipeline_index < len(self.pipeline_specs)
                and result[smallest_non_zero_pipeline_index] == 0
            ):
                smallest_non_zero_pipeline_index += 1

            if (
                smallest_non_zero_pipeline_index + 1 < len(self.pipeline_specs)
                and result[smallest_non_zero_pipeline_index] > 0
            ):
                result[smallest_non_zero_pipeline_index] -= 1
                result[smallest_non_zero_pipeline_index + 1] += 1
                total_assigned_nodes += 1

        assert (
            sum(
                result[i] * self.pipeline_specs[i].num_nodes
                for i in range(0, len(self.pipeline_specs))
            )
            == num_nodes
        )

        return result
