from typing import List

import pytest
import torch

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
)
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)
from tests.conftest import OobleckSingleProcessTestCase


class TestPipelineInstantiator(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def profile(self):
        return self.factory.get_dummy_profile()

    def test_create_pipelines_one_template(self, profile: LayerExecutionResults):
        # Get a dummy template for 1 GPU and instantiate pipelines with it.
        # Expected result is to have 4 identical pipelines instantiated from the template.
        template = self.factory.get_dummy_pipeline_template(num_gpus=1)
        allreduce_across_nodes = [
            profile.at(i)._allreduce_across_nodes for i in range(profile.size)
        ]
        instantiator = PipelineInstantiator()
        execution_plan = instantiator.get_best_execution_plan(
            pipeline_templates=[template],
            allreduce_across_nodes=allreduce_across_nodes,
            num_nodes=4,
            global_num_microbatch=512,
        )
        assert isinstance(execution_plan, HeterogeneousPipelinesExecutionPlan)
        assert len(execution_plan.pipeline_templates) == 1
        assert execution_plan.pipeline_templates[0] == template
        assert execution_plan.num_instances_set[template] == 4
        assert (
            execution_plan.num_instances_set[template]
            * execution_plan.num_microbatches_set[template]
            == 512
        )

    def test_create_pipelines_multiple_templates(self, profile: LayerExecutionResults):
        num_nodes = 13
        global_batch = 1024

        templates: List[PipelineTemplate] = [
            self.factory.get_dummy_pipeline_template(num_gpus=i) for i in range(2, 8)
        ]
        allreduce_across_nodes = [
            profile.at(i)._allreduce_across_nodes for i in range(profile.size)
        ]
        instantiator = PipelineInstantiator()
        execution_plan = instantiator.get_best_execution_plan(
            pipeline_templates=templates,
            allreduce_across_nodes=allreduce_across_nodes,
            num_nodes=num_nodes,
            global_num_microbatch=global_batch,
        )
        assert isinstance(execution_plan, HeterogeneousPipelinesExecutionPlan)
        assert 1 < len(execution_plan.pipeline_templates) < 8
        assert all(t in templates for t in execution_plan.pipeline_templates)
        assert (
            sum(
                pipeline_template._num_nodes
                * pipeline_template._num_gpus_per_node
                * num_instances
                for pipeline_template, num_instances in execution_plan.num_instances_set.items()
            )
            == num_nodes
        )
        assert all(mb > 0 for mb in execution_plan.num_microbatches_set.values())
        assert (
            sum(
                execution_plan.num_instances_set[t]
                * execution_plan.num_microbatches_set[t]
                for t in execution_plan.pipeline_templates
            )
            == global_batch
        )

    @pytest.mark.skip(reason="Not implemented yet")
    def test_create_pipelines_multiple_templates_multigpu_per_node(
        self, profile: LayerExecutionResults
    ):
        pass
