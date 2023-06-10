from typing import List, Tuple
from oobleck.csrc.planning.execution_result import (
    LayerExecutionResults,
    StageExecutionResult,
)

def get_profiler_results(
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
