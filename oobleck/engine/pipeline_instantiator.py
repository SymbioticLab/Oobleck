from oobleck.planning.pipeline_template import PipelineTemplate


class PipelineInstantiator:
    """A class that determines the number of pipelines to be instantiated
    from each pipeline template and the number of microbatches.
    """

    def __init__(self, pipeline_templates: list[PipelineTemplate]):
        self.pipeline_templates = pipeline_templates
        self.instantiated_pipelines = None

    def instantiate(
        self, world_size: int
    ) -> tuple[dict[PipelineTemplate, int], dict[PipelineTemplate, int]]:
        """Instantiate pipelines from pipeline templates.
        If there are already instantiated pipelines, this method will reconfigure
        pipelines for a new distributed configuration.
        """
        pass
