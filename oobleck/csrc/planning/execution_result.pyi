from typing import List, Dict, Tuple

class LayerExecutionResult:
    def __init__(
        self,
        layer_index: int,
        forward: float,
        backward: float,
        allreduce_in_node: Dict[int, float],
        allreduce_across_nodes: Dict[int, float],
        mem_required: Tuple[int, int],
    ): ...
    _layer_index: int
    _forward: float
    _backward: float
    _allreduce_in_node: Dict[int, float]
    _allreduce_across_nodes: Dict[int, float]
    _mem_required: Tuple[int, int]

class LayerExecutionResults:
    def get(self, index: int) -> LayerExecutionResult: ...
    def size(self) -> int: ...

class StageExecutionResult:
    _num_gpus: int
    _layer_indices: List[int]
    _forward: float
    _backward: float
    _allreduce_cross_nodes: Dict[int, float]
    _mem_required: int
