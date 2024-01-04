import pytest

from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck_colossalai.pipeline_template import PipelineTemplate

from conftest import homogeneous_templates, heterogeneous_templates


@pytest.mark.parametrize(
    "templates",
    [homogeneous_templates, heterogeneous_templates],
    ids=["homogeneous", "heterogeneous"],
)
def test_instantiate(templates: dict[PipelineTemplate, int]):
    instantiator = PipelineInstantiator(list(templates.keys()), 512)
    result = instantiator.instantiate(6)

    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)
    assert isinstance(result[1], dict)

    total_num_microbatches = sum(
        result[0][template] * num_mb for template, num_mb in result[1].items()
    )
    assert total_num_microbatches == 512
    total_num_nodes = sum(
        template.num_stages * num_pipelines
        for template, num_pipelines in result[0].items()
    )
    assert total_num_nodes == 6
