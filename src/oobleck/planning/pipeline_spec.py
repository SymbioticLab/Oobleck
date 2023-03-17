import math

from typing import List, Iterator
from deepspeed.utils.logging import logger

from oobleck.module.model import OobleckModel


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

        self.optimal_plan = self.get_optimal_execution_plan()

    def __repr__(self) -> str:
        return f"(PipelineSpec: {self.num_nodes} nodes)"

    def get_optimal_execution_plan(self) -> List[int]:
        """Oobleck paper section 4.1.2. Optimal Execution Plan implementation
        It partitions the model into stages and assigns layers into them,
        and map the stages with a group of GPUs that provides the highest throughput.

        This currently does not consider intra-node parallelism.
        For intra-node parallelism, each rank group must create its own :class:`PipelineSpec`.

        Returns:
            List[int]: The list of ranks that each layer is assigned to.
        """
        num_layer_per_node = len(self.model.model) // self.num_nodes
        # num_layers: number of layers that are assigned to each stage.
        num_layers = [num_layer_per_node] * self.num_nodes
        if num_layer_per_node * self.num_nodes < len(self.model.model):
            num_layers[-1] += (
                len(self.model.model) - num_layer_per_node * self.num_nodes
            )

        layer_specs: List[int] = []
        sum = 0
        for i, (node_id, num_layer) in enumerate(
            zip(range(self.num_nodes), num_layers)
        ):
            end_index = sum + num_layer
            for _ in range(sum, end_index):
                layer_specs.append(i * node_id)
            sum += num_layer

        return layer_specs
