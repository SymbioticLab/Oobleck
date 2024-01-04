from __future__ import annotations

from transformers import PretrainedConfig


class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines."""

    def __init__(
        self,
        model: PretrainedConfig,
        num_nodes: int,
        gpus_per_node: int,
        modules_per_stage: list[list[str]],
    ):
        self.model = model
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.modules_per_stage = modules_per_stage

    @property
    def num_layers(self) -> int:
        return sum(len(stage) for stage in self.modules_per_stage)

    @property
    def num_stages(self) -> int:
        return len(self.modules_per_stage)

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @staticmethod
    def generate_pipeline_templates() -> list[PipelineTemplate]:
        raise NotImplementedError
