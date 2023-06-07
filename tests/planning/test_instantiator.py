import pytest

from oobleck.planning.instantiator import PipelineInstantiator


@pytest.mark.order(after="test_profiler.py::test_profile")
def test_initialize_instantiator(gpt2_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (2, 5),  # num nodes range
        1,
    )

    instantiator = PipelineInstantiator()
    execution_plan = instantiator.get_best_execution_plan(
        model_layers=pipeline_template_generator.get_profiler_results(),
        pipeline_templates=pipeline_templates,
        num_nodes=13,
        global_num_microbach=512,
    )
