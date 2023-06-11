import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
)


def test_create_pipeline_templates_onenode(
    dummy_layer_execution_results: LayerExecutionResults,
):
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        dummy_layer_execution_results,
        (1, 1),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0]._num_nodes == 1
    assert pipeline_templates[0]._num_gpus_per_node == 1
    assert len(pipeline_templates[0]._stages) == 1
    assert pipeline_templates[0]._iteration_time > 0


def test_create_pipeline_templates_maxnode(
    dummy_layer_execution_results: LayerExecutionResults,
):
    generator = PipelineTemplateGenerator()
    num_nodes = dummy_layer_execution_results._size
    pipeline_templates = generator.create_pipeline_templates(
        dummy_layer_execution_results,
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0]._num_nodes == num_nodes
    assert pipeline_templates[0]._num_gpus_per_node == 1
    assert len(pipeline_templates[0]._stages) == num_nodes
    assert pipeline_templates[0]._iteration_time > 0


def test_create_pipeline_templates_toomanynodes(
    dummy_layer_execution_results: LayerExecutionResults,
):
    generator = PipelineTemplateGenerator()
    num_nodes = dummy_layer_execution_results._size + 1
    pipeline_templates = generator.create_pipeline_templates(
        dummy_layer_execution_results,
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 0


def test_create_pipeline_templates_noderange(
    dummy_layer_execution_results: LayerExecutionResults,
):
    generator = PipelineTemplateGenerator()
    num_nodes = dummy_layer_execution_results._size
    pipeline_templates = generator.create_pipeline_templates(
        dummy_layer_execution_results,
        (1, num_nodes),  # num nodes range
        1,
    )
    assert 0 < len(pipeline_templates) <= num_nodes
    assert 0 < pipeline_templates[0]._num_nodes <= num_nodes
    assert len(pipeline_templates[0]._stages) == 1
    assert len(pipeline_templates[-1]._stages) == num_nodes
    for pipeline_template in pipeline_templates:
        assert pipeline_templates[0]._num_gpus_per_node == 1
        assert len(pipeline_template._stages) > 0
        assert pipeline_template._iteration_time > 0


@pytest.mark.skip(reason="TODO")
def test_create_pipeline_templates_fsdp(model):
    assert False
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (2, 10),  # num nodes range
        4,
    )
