from typing import Dict, List, Tuple

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
    _index: int
    _forward: float
    _backward: float
    _allreduce_in_node: Dict[int, float]
    _allreduce_across_nodes: Dict[int, float]
    _mem_required: Tuple[int, int]

class LayerExecutionResults:
    def get(self) -> List[LayerExecutionResult]: ...
    def at(self, index: int) -> LayerExecutionResult: ...
    def size(self) -> int: ...

class StageExecutionResult:
    _num_gpus: int
    _layer_indices: List[int]
    _num_layers: int
    _mem_required: int

def get_profile_results(
    model_name: str, model_tag: str, microbatch_size: int
) -> LayerExecutionResults: ...

class PipelineTemplate:
    def __init__(
        self,
        stages: List[StageExecutionResult],
        iteration_time: float,
        num_layers: int,
        num_nodes: int,
        num_gpus_per_node: int,
    ): ...
    def get_iteration_time(self) -> float: ...
    def get_stages(self) -> List[StageExecutionResult]: ...
    def get_num_nodes(self) -> int: ...
    def get_num_gpus_per_node(self) -> int: ...

class PipelineTemplateGenerator:
    def __init__(self): ...
    def create_pipeline_templates(
        self,
        layer_execution_results: LayerExecutionResults,
        num_nodes: Tuple[int, int],
        num_gpus_per_node: int,
    ) -> List[PipelineTemplate]: ...
