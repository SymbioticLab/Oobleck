import logging
import time

import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
)


# `ft_threshold` indicates fault tolerance thresholds. Oobleck should maintain at least ftt + 1 copies.
@pytest.mark.parametrize("ft_threshold", [1, 3], ids=["ft1", "ft3"])
# `min_num_nodes` indicates the minimum number of nodes fow model traning.
@pytest.mark.parametrize("min_num_nodes", [2, 3], ids=["min2", "min3"])
def test_generate_pipeline_template(
    logger: logging.Logger,
    eval_model_name: str,
    ft_threshold: int,
    min_num_nodes: int,
    eval_dummy_profile: LayerExecutionResults,
):
    # initial number of nodes is 13.
    num_nodes = 13

    # n^{max}_{p-1} = N - f0
    max_num_nodes_per_pipeline = num_nodes - ft_threshold

    logger.info(
        f"""PIPELINE TEMPLATE GENERATION for {eval_model_name}
        Layer execution times are {"constant to 2.5s" if eval_model_name == "fake_model1" else "random (0~5)s"}
        Config - ft_threshold: {ft_threshold}, minimum number of nodes for training (n0): {min_num_nodes}, number of nodes: {num_nodes}
        This configuration guarantees to run training with {(ft_threshold + 1) * min_num_nodes} ~ {num_nodes} nodes.
        Expected number of pipeline templates: {max_num_nodes_per_pipeline - min_num_nodes + 1} (pipelines with number of nodes {list(range(min_num_nodes, max_num_nodes_per_pipeline + 1))})
        ==============================
        """
    )

    start = time.time_ns()
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        eval_dummy_profile,
        (min_num_nodes, max_num_nodes_per_pipeline),
        1,
    )
    end = time.time_ns()

    # Check number of generated pipelines is correct.
    expected_num_pipeline_templates = max_num_nodes_per_pipeline - min_num_nodes + 1
    logger.info(
        f"Generated number of pipelines: {len(pipeline_templates)} (expected: {expected_num_pipeline_templates})"
    )
    assert len(pipeline_templates) == expected_num_pipeline_templates

    logger.info("Checking if pipeline stages are balanced...")
    for index, pipeline_template in enumerate(pipeline_templates):
        stages = pipeline_template.get_stages()
        message = f"    [template {index}] {len(stages)} stages: "
        for stage in stages:
            message += f"[{(stage._forward + stage._backward):.1f} ({len(stage._layer_indices)} layers)] "
        logger.info(message)
        assert len(stages) == index + min_num_nodes

    logger.info(
        "Pipeline templates generation time using divide and conquer: %.2f ms"
        % ((end - start) / 1e6)
    )
