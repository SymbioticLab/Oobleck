import pytest
import torch

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
    assert execution_plan is not None
    # execution_plan.pipeline_templates may have less templates
    # if they are not used.
    assert all(
        template in pipeline_templates for template in execution_plan.pipeline_templates
    )
    assert all(
        template in execution_plan.pipeline_templates
        for template in execution_plan.num_instances_set.keys()
    )
    assert all(
        template in execution_plan.pipeline_templates
        for template in execution_plan.num_microbatches_set.keys()
    )
    # Check all nodes are used
    assert (
        sum(
            template.get_num_nodes() * num_templates
            for template, num_templates in execution_plan.num_instances_set.items()
        )
        == 13
    )
    # Check global batch size is correct
    assert (
        sum(
            execution_plan.num_instances_set[template]
            * execution_plan.num_microbatches_set[template]
            for template in execution_plan.pipeline_templates
        )
        == 512
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_initialize_instantiator_multigpu():
    assert False, "Not implemented yet"
