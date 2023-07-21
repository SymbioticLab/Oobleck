import gc
import logging
import time

import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
)
from oobleck.planning.instantiator import PipelineInstantiator


def eval_dummy_pipeline_templates(
    min_num_nodes: int,
    eval_dummy_profile: LayerExecutionResults,
) -> list[PipelineTemplate]:
    generator = PipelineTemplateGenerator()
    # Use 7 pipeline templates for evaluation
    return generator.create_pipeline_templates(
        eval_dummy_profile,
        (min_num_nodes, min_num_nodes + 6),
        1,
    )


# @pytest.mark.forked
@pytest.mark.parametrize(
    "num_nodes", [13, 16, 21], ids=["13nodes", "16nodes", "21nodes"]
)
# NOTE: should be large enough so that distribution can happen.
# We can release this constraint but current implementation is a strict version.
@pytest.mark.parametrize("global_num_microbatches", [512, 4096])
def test_instantiate_pipelines(
    logger: logging.Logger,
    num_nodes: int,
    global_num_microbatches: int,
    eval_model_name: str,
    eval_dummy_profile: LayerExecutionResults,
):
    # `ft_threshold` indicates fault tolerance thresholds. Oobleck should maintain at least ftt + 1 copies.
    ft_threshold: int = 3
    # `min_num_nodes` indicates the minimum number of nodes fow model traning.
    min_num_nodes: int = 3

    pipeline_templates = eval_dummy_pipeline_templates(
        min_num_nodes, eval_dummy_profile
    )

    logger.info(
        f"""PIPELINE INSTANTIATION for {eval_model_name}
        Config - ft_threshold: {ft_threshold}, minimum number of nodes for training (n0): {min_num_nodes}, number of nodes: {num_nodes}
        Number of pipeline templates: {len(pipeline_templates)} (pipelines with number of nodes {[template._num_nodes for template in pipeline_templates]})
        """
    )
    for template_id, template in enumerate(pipeline_templates):
        logger.info(f"[template ID: {template_id}] {len(template.get_stages())} stages")

    allreduce_across_nodes = [
        eval_dummy_profile.at(i)._allreduce_across_nodes
        for i in range(eval_dummy_profile.size)
    ]

    start = time.time_ns()
    instantiator = PipelineInstantiator()
    execution_plan = instantiator.get_best_execution_plan(
        pipeline_templates=pipeline_templates,
        allreduce_across_nodes=allreduce_across_nodes,
        num_nodes=num_nodes,
        global_num_microbatch=global_num_microbatches,
    )
    end = time.time_ns()

    msg = ""
    for template, num_instances in execution_plan.num_instances_set.items():
        num_microbatches = execution_plan.num_microbatches_set[template]
        # Dividing time by number of stages is due to the fact that estimating iteration time relied on the number of stages.
        msg += (
            f"[template ID {pipeline_templates.index(template)} - {template._num_nodes} nodes]: "
            f"{num_instances} instances (total {num_instances * template._num_nodes} nodes assigned). "
            f"Each pipeline has {num_microbatches} number of microbatches "
            f"(estimated pipeline execution time: {(template._iteration_time * num_microbatches / len(template.get_stages())):.2f} ms).\n"
        )

    total_num_nodes_used = sum(
        pt._num_nodes * num_instances
        for pt, num_instances in execution_plan.num_instances_set.items()
    )
    total_num_microbatches = sum(
        execution_plan.num_instances_set[pt] * execution_plan.num_microbatches_set[pt]
        for pt in execution_plan.pipeline_templates
    )
    logger.info(
        f"""PIPELINE INSTANTIATION RESULT
{msg}
        Total number of nodes used: {total_num_nodes_used} (expected: {num_nodes})
        Total number of microbatches: {total_num_microbatches} (expected: {global_num_microbatches})
        """
    )
    assert total_num_nodes_used == num_nodes
    assert total_num_microbatches == global_num_microbatches

    logger.info(
        "Pipeline instantiation time using dynamic programming and linear optimization: %.2f ms"
        % ((end - start) / 1e6)
    )
    gc.collect()
