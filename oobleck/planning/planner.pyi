from typing import Any

from oobleck.planning.profiler import LayerExecutionResult


def create_pipeline_templates(
    model_name: str,
    profile_data: list[LayerExecutionResult],
    resource_counts: list[int],
    tensor_parallel_size: int = 1,
    device_memory_bytes: int | None = None,
) -> dict[int, dict[str, Any]]: ...
