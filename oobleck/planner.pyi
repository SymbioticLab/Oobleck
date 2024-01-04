from pathlib import Path

class PipelineTemplate:
    latency: float
    mem_required: int
    modules_per_stage: list[list[str]]

def create_pipeline_templates(
    self, model_name: str, tag: str, num_nodes: list[int], oobleck_base_dir: Path
) -> dict[int, PipelineTemplate]: ...
