from typing import List

import pytest
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.distributed_utils import (
    create_mock_pg,
    mock_init_dist,
)

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
)
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)
from tests.conftest import (
    TRAIN_BATCH_SIZE,
    OobleckDynamicClassFactory,
    OobleckSingleProcessTestCase,
    OobleckStaticClassFactory,
)


class TestPipelineInstantiator(OobleckSingleProcessTestCase):
    factory: OobleckStaticClassFactory

    @pytest.fixture(scope="function")
    def profile(self):
        return self.factory.get_dummy_profile()

    @pytest.mark.parametrize("num_gpus", [1, 2, 4])
    def test_create_pipelines_one_template(
        self, num_gpus: int, profile: LayerExecutionResults
    ):
        # Get a dummy template for 1 GPU and instantiate pipelines with it.
        # Expected result is to have 4 identical pipelines instantiated from the template.
        template = self.factory.get_dummy_pipeline_template(num_gpus=num_gpus)
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
        assert execution_plan.num_instances_set[template] == 4 // num_gpus
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

    @pytest.mark.parametrize("num_gpus", [1, 2, 4], ids=lambda x: f"{x} GPUs")
    @pytest.mark.parametrize("rank", [0, 5, 10, 15], ids=lambda x: f"rank{x}")
    def test_pipeline_rank_index(
        self,
        num_gpus: int,
        rank: int,
        profile: LayerExecutionResults,
        mocker: MockerFixture,
    ):
        world_size = 16
        templates: PipelineTemplate = self.factory.get_dummy_pipeline_template(
            num_gpus=num_gpus
        )
        allreduce_across_nodes = [l._allreduce_across_nodes for l in profile.get()]
        instantiator = PipelineInstantiator()
        plan = instantiator.get_best_execution_plan(
            pipeline_templates=[templates],
            allreduce_across_nodes=allreduce_across_nodes,
            num_nodes=world_size,
            global_num_microbatch=512,
        )

        total_pipeline_num = world_size // num_gpus
        target_pipeline_index: int = rank // num_gpus

        model: OobleckModel = self.factory.get_model()

        # mocker.patch(
        #     "torch.distributed.init_process_group",
        #     return_value=mock_init_dist(rank, world_size),
        # )
        mock_init_dist(rank, world_size)
        mocker.patch("deepspeed.comm.is_initialized", return_value=True)
        mocker.patch("deepspeed.comm.get_rank", return_value=rank)
        mocker.patch(
            "torch.distributed.new_group",
            return_value=create_mock_pg(None, rank, world_size, None),
        )

        dfactory = OobleckDynamicClassFactory(
            static_factory=self.factory,
            my_rank=rank,
            ranks=list(range(0, 16)),
        )

        pipeline: OobleckPipeline = plan.instantiate(
            model=model,
            dataloader=dfactory.get_dataloader(
                pipeline_index=target_pipeline_index,
                num_microbatches=[TRAIN_BATCH_SIZE] * total_pipeline_num,
            ),
            training_args=self.factory._training_args,
        )

        # Fix it...
        # assert pipeline.pipeline_id == target_pipeline_index
        # assert pipeline.my_rank == rank
        # assert len(pipeline.ranks) == num_gpus
