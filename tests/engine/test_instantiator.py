import pytest
from conftest import singlenode_template, homogeneous_templates, heterogeneous_templates

from oobleck.engine.pipeline_instantiator import PipelineInstantiator


@pytest.mark.parametrize("num_nodes", list(range(4, 17)))
def test_instantiate(num_nodes: int):
    instantiator = PipelineInstantiator(list(heterogeneous_templates.keys()), 512)
    result = instantiator.instantiate(num_nodes)

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
    assert total_num_nodes == num_nodes


def test_instantiate_onenode():
    instantiator = PipelineInstantiator(list(singlenode_template.keys()), 32)
    result = instantiator.instantiate(1)

    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)
    assert isinstance(result[1], dict)

    total_num_microbatches = sum(
        result[0][template] * num_mb for template, num_mb in result[1].items()
    )
    assert total_num_microbatches == 32
    total_num_nodes = sum(
        template.num_stages * num_pipelines
        for template, num_pipelines in result[0].items()
    )
    assert total_num_nodes == 1
