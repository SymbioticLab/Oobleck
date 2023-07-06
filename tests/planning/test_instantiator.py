import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
)
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)
from tests.conftest import OobleckSingleProcessTestCase


@pytest.mark.parametrize(
    "num_gpus_per_node", [1, 2, 4], ids=["1gpu/node", "2gpus/node", "4gpus/node"]
)
class TestPipelineInstantiator(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def profile(self):
        return self.factory.get_dummy_profile()

    def test_create_pipelines_one_template(
        self, profile: LayerExecutionResults, num_gpus_per_node: int
    ):
        # Get a dummy template for 1 GPU and instantiate pipelines with it.
        # Expected result is to have 4 identical pipelines instantiated from the template.
        template = self.factory.get_dummy_pipeline_template(
            num_stages=1, num_nodes=1, num_gpus_per_node=num_gpus_per_node
        )
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

    def test_create_pipelines_multiple_templates(
        self, profile: LayerExecutionResults, num_gpus_per_node: int
    ):
        num_nodes = 13
        global_batch = 1024

        templates: list[PipelineTemplate] = [
            self.factory.get_dummy_pipeline_template(
                num_stages=i, num_nodes=i, num_gpus_per_node=num_gpus_per_node
            )
            for i in range(2, 8)
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
                pipeline_template._num_nodes * num_instances
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

    def test_create_pipelines_multinodes(
        self, profile: LayerExecutionResults, num_gpus_per_node: int
    ):
        num_nodes = 4
        template: PipelineTemplate = self.factory.get_dummy_pipeline_template(
            num_stages=4, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
        )
        allreduce_across_nodes = [
            profile.at(i)._allreduce_across_nodes for i in range(profile.size)
        ]
        instantiator = PipelineInstantiator()
        execution_plan = instantiator.get_best_execution_plan(
            pipeline_templates=[template],
            allreduce_across_nodes=allreduce_across_nodes,
            num_nodes=num_nodes,
            global_num_microbatch=512,
        )
        assert isinstance(execution_plan, HeterogeneousPipelinesExecutionPlan)
        assert execution_plan.pipeline_templates[0] == template
        assert (
            execution_plan.pipeline_templates[0]._num_gpus_per_node == num_gpus_per_node
        )
        assert execution_plan.pipeline_templates[0]._num_nodes == num_nodes
        assert (
            sum(
                stage._num_gpus
                for stage in execution_plan.pipeline_templates[0]._stages
            )
            == num_nodes * num_gpus_per_node
        )
        assert (
            sum(
                len(stage._layer_indices)
                for stage in execution_plan.pipeline_templates[0]._stages
            )
            == profile.size
        )
