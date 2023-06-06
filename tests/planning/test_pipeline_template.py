import pytest


@pytest.mark.order(after="test_profiler.py::test_profile")
def test_create_pipeline_templates_onenode(gpt2_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (1, 1),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0].get_num_nodes() == 1
    assert pipeline_templates[0].get_num_gpus_per_node() == 1
    assert len(pipeline_templates[0].get_stages()) == 1
    assert pipeline_templates[0].get_iteration_time() > 0


@pytest.mark.order(after="test_profiler.py::test_profile")
def test_create_pipeline_templates_maxnode(gpt2_model, pipeline_template_generator):
    num_nodes = len(gpt2_model.model)
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0].get_num_nodes() == num_nodes
    assert pipeline_templates[0].get_num_gpus_per_node() == 1
    assert len(pipeline_templates[0].get_stages()) == num_nodes
    assert pipeline_templates[0].get_iteration_time() > 0


@pytest.mark.order(after="test_profiler.py::test_profile")
def test_create_pipeline_templates_toomanynodes(
    gpt2_model, pipeline_template_generator
):
    num_nodes = len(gpt2_model.model) + 1
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 0


@pytest.mark.order(after="test_profiler.py::test_profile")
def test_create_pipeline_templates_noderange(gpt2_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (1, len(gpt2_model.model)),  # num nodes range
        1,
    )
    assert 0 < len(pipeline_templates) <= len(gpt2_model.model)
    assert 0 < pipeline_templates[0].get_num_nodes() <= len(gpt2_model.model)
    assert len(pipeline_templates[0].get_stages()) == 1
    assert len(pipeline_templates[-1].get_stages()) == len(gpt2_model.model)
    for pipeline_template in pipeline_templates:
        assert pipeline_templates[0].get_num_gpus_per_node() == 1
        assert len(pipeline_template.get_stages()) > 0
        assert pipeline_template.get_iteration_time() > 0


@pytest.mark.order(after="test_profiler.py::test_profile_multimicrobatch")
def test_create_pipeline_templates_multimicrobatch(
    gpt2_model, pipeline_template_generator
):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        4,  # microbatch_size
        (1, len(gpt2_model.model)),  # num nodes range
        1,
    )
    assert 0 < len(pipeline_templates) <= len(gpt2_model.model)
    assert 0 < pipeline_templates[0].get_num_nodes() <= len(gpt2_model.model)
    assert len(pipeline_templates[0].get_stages()) == 1
    assert len(pipeline_templates[-1].get_stages()) == len(gpt2_model.model)
    for pipeline_template in pipeline_templates:
        assert pipeline_templates[0].get_num_gpus_per_node() == 1
        assert len(pipeline_template.get_stages()) > 0
        assert pipeline_template.get_iteration_time() > 0


@pytest.mark.skip(reason="TODO")
def test_create_pipeline_templates_fsdp(gpt2_model, pipeline_template_generator):
    assert False
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (2, 10),  # num nodes range
        4,
    )
