import pytest
from conftest import template_1stage, template_2stages, template_3stages

from oobleck.engine.pipeline_instantiator import PipelineInstantiator


@pytest.mark.parametrize(
    "num_nodes",
    list(range(4, 17)),
    ids=lambda num_nodes: f"{num_nodes} nodes",
)
def test_instantiate(num_nodes: int):
    instantiator = PipelineInstantiator([template_2stages, template_3stages], 512)
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
    instantiator = PipelineInstantiator([template_1stage], 32)
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
