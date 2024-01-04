from pathlib import Path
from oobleck_colossalai.pipeline_template import PipelineTemplate

def create_pipeline_templates(
    tag: str, num_nodes: list[int], oobleck_base_dir: Path
) -> dict[int, PipelineTemplate]: ...
