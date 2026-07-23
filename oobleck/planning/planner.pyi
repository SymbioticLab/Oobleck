from oobleck.planning.profiler import LayerExecutionResult

def create_pipeline_templates(
    model_name: str,
    profile_data: list[LayerExecutionResult],
    num_nodes: list[int],
    tensor_parallel_size: int = 1,
) -> dict[int, dict]: ...
