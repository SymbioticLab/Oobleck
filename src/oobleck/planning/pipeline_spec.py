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

        self.layers_spec = self._create_optimal_plan(model)

    def _create_optimal_plan(self, model: OobleckModel) -> List[int]:
        """Create an optimal execution plan with the given number of GPUs
        using profiled model execution information.

        Current Alpha-level implementation: divide layers individually.
        Number of stages is equal to the number of nodes.

        This currently does not consider intra-node parallelism.
        For intra-node parallelism, each rank group must create its own :class:`PipelineSpec`.

        Args:
            model (OobleckModel): model to profile.

        Returns:
            List[int]: The list of ranks that are assigned to each layer in the model.
        """
        num_layer_per_node = len(model.model) // self.num_nodes
        # num_layers: number of layers that are assigned to each stage.
        num_layers = [num_layer_per_node] * self.num_nodes
        if num_layer_per_node * self.num_nodes < len(model.model):
            num_layers[-1] += len(model.model) - num_layer_per_node * self.num_nodes

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


def get_pipeline_specs(
    ft_spec: int,
    min_num_nodes: int,
    max_num_nodes: int,
    num_gpus_per_node: int,
    model: OobleckModel,
) -> List[PipelineSpec]:
    """Generates the list of :class:`.PipelineSpec`s that can represent any number N that
    min_num_nodes <= N <= max_num_nodes as a linear combination of them.

    Args:
        ft_spec (int): Fault tolerant spec.
            Oobleck tries to create at least ft_spec + 1 model replica.
        min_num_nodes (int): Mininum # nodes to hold the model states for training.
        max_num_nodes (int): Maximum # nodes in the cluster.
        num_gpus_per_node (int): # GPUs per node.

    Raises:
        ValueError: Raised when given argument is unreasonable.
    """
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

    pipeline_specs = [
        PipelineSpec(num_nodes, num_gpus_per_node, model)
        for num_nodes in pipeline_spec_num_nodes
    ]

    return pipeline_specs
