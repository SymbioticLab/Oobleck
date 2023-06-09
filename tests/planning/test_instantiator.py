import pytest

from oobleck.planning.instantiator import PipelineInstantiator


def test_initialize_instantiator(
    gpt2_model, gpt2_dummy_profile_results, pipeline_template_generator
):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (2, 5),  # num nodes range
        1,
    )

    instantiator = PipelineInstantiator()
    execution_plan = instantiator.get_best_execution_plan(
        pipeline_templates=pipeline_templates,
        allreduce_across_nodes=[
            result.allreduce_cross_nodes for result in gpt2_dummy_profile_results
        ],
        num_nodes=13,
        global_num_microbatch=512,
    )
