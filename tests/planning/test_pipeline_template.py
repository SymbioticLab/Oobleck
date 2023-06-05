import pytest


def test_create_pipeline_templates_onenode(gpt2_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (1, 1),  # num nodes range
        1,
    )
    assert len(pipeline_templates) <= 1


def test_create_pipeline_templates_manynodes(gpt2_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (1, len(gpt2_model.model)),  # num nodes range
        1,
    )
    assert len(pipeline_templates) <= len(gpt2_model.model)


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
