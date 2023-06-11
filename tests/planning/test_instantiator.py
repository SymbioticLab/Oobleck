from typing import List

import pytest
import torch

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_instantiate_pipelines_single_template(
    dummy_profile_results, dummy_pipeline_template
):
    allreduce_across_nodes = dummy_profile_results[1]
    pipeline_template: PipelineTemplate = dummy_pipeline_template(num_gpus=1)
    instantiator = PipelineInstantiator()
    execution_plan = instantiator.get_best_execution_plan(
        pipeline_templates=[pipeline_template],
        allreduce_across_nodes=allreduce_across_nodes,
        num_nodes=4,
        global_num_microbatch=512,
    )
    assert isinstance(execution_plan, HeterogeneousPipelinesExecutionPlan)
    assert len(execution_plan.pipeline_templates) == 1
    assert execution_plan.pipeline_templates[0] == pipeline_template
    assert execution_plan.num_instances_set[pipeline_template] == 4
    assert (
        execution_plan.num_instances_set[pipeline_template]
        * execution_plan.num_microbatches_set[pipeline_template]
        == 512
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_instantiate_pipelines_multiple_templates(
    dummy_profile_results, dummy_pipeline_template
):
    allreduce_across_nodes = dummy_profile_results[1]
    pipeline_templates: List[PipelineTemplate] = [
        dummy_pipeline_template(num_gpus=i) for i in range(2, 6)
    ]
    instantiator = PipelineInstantiator()
    execution_plan = instantiator.get_best_execution_plan(
        pipeline_templates=pipeline_templates,
        allreduce_across_nodes=allreduce_across_nodes,
        num_nodes=13,  # no single pipeline template can cover this # nodes.
        global_num_microbatch=512,
    )
    assert isinstance(execution_plan, HeterogeneousPipelinesExecutionPlan)
    assert 1 < len(execution_plan.pipeline_templates) < 6
    assert all(t in pipeline_templates for t in execution_plan.pipeline_templates)
    assert (
        sum(
            pipeline_template._num_nodes
            * pipeline_template._num_gpus_per_node
            * num_instances
            for pipeline_template, num_instances in execution_plan.num_instances_set.items()
        )
        == 13
    )
    assert all(mb > 0 for mb in execution_plan.num_microbatches_set.values())
    assert (
        sum(
            execution_plan.num_instances_set[t] * execution_plan.num_microbatches_set[t]
            for t in execution_plan.pipeline_templates
        )
        == 512
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_initialize_instantiator_multigpu():
    # Test if each node has more than 1 GPU.
    assert False, "Not implemented yet"
