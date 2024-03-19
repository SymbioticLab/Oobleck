from pathlib import Path

from oobleck_colossalai.pipeline_template import PipelineTemplate

def create_pipeline_templates(
    model_name: str, microbatch_size: int, num_nodes: list[int], job_profile_dir: Path
) -> dict[int, PipelineTemplate]: ...
