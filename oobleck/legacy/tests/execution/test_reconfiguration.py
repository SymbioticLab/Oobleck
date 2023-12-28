from __future__ import annotations

import logging
from multiprocessing import connection
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.engine import ReconfigurationEngine
from oobleck.execution.pipeline import OobleckPipeline
from tests.conftest import OobleckSingleProcessTestCase


class FakeExecution:
    _layers: list[None] = []


class FakePipeline:
    def __init__(
        self,
        pipeline_id: int,
        pipeline_template: PipelineTemplate,
        ranks: list[int],
        dataloader: OobleckDataLoader,
        *args,
        **kwargs,
    ):
        self._pipeline_id = pipeline_id
        self._template = pipeline_template
        self._ranks = ranks
        self._dataloader = dataloader
        self._global_step = 0
        self.execution = FakeExecution()
        self.my_pipeline = bool(0 in self._ranks)

        # copy from pipeline __init__
        # layer_index -> ranks
        self.rank_grid: dict[int, list[int]] = pipeline_template.get_rank_grid(
            self._ranks
        )

    def initialize_distributed_fsdp(self, *args):
        pass

    def initialize_distributed_pipeline(self, *args):
        pass

    def initialize_execution(self, *args):
        pass


class FakeProcessGroup:
    def __init__(self, ranks: list[int]):
        self.ranks = ranks


class FakeEngine:
    def __init__(
        self,
        test_class: TestReconfigurationClass,
        sample_args: OobleckArguments,
    ):
        self.test_class = test_class
        self._agent_pipe: connection.Connection = None
        self._args = sample_args
        self._hf_training_args = test_class.factory._training_args
        self._num_nodes = sum(range(2, 6))
        self._dataset = test_class.factory.get_dataset()
        self._model = test_class.factory.get_model()
        self._profile_results = test_class.factory.get_dummy_profile()

    def init_pipelines(self, num_gpus_per_node: int) -> list[OobleckPipeline]:
        self._num_gpus_per_node = num_gpus_per_node
        self._pipeline_templates = [
            self.test_class.factory.get_dummy_pipeline_template(
                num_stages=i,
                num_gpus_per_node=num_gpus_per_node,
                num_nodes=i,
            )
            for i in range(2, 6)
        ]

        num_gpus_used: int = 0
        pipeline_id: int = 0
        pipelines: list[OobleckPipeline] = []

        dataloader = OobleckDataLoader(
            self._hf_training_args, self._dataset, LoaderType.Training, 0, [0], 0, 0
        )

        # Have one pipelines for each pipeline template
        for template in self._pipeline_templates:
            num_gpus_per_pipeline = sum(
                [stage._num_gpus for stage in template.get_stages()]
            )
            pipelines.append(
                FakePipeline(
                    pipeline_id,
                    template,
                    list(
                        range(
                            num_gpus_used,
                            num_gpus_used + num_gpus_per_pipeline,
                        )
                    ),
                    dataloader,
                )
            )

            pipeline_id += 1
            num_gpus_used += num_gpus_per_pipeline
        self._pipeline = pipelines[0]
        return pipelines


class TestReconfigurationClass(OobleckSingleProcessTestCase):
    @pytest.fixture
    def fake_engine(
        self, mocker: MockerFixture, sample_args: OobleckArguments
    ) -> FakeEngine:
        mocker.patch(
            "oobleck.planning.instantiator.OobleckPipeline",
            new=FakePipeline,
        )
        mocker.patch(
            "deepspeed.comm.new_group",
            new=FakeProcessGroup,
        )
        mocker.patch(
            "deepspeed.comm.get_rank",
            return_value=0,
        )
        mocker.patch(
            "torch.distributed.init_process_group",
            new=FakeProcessGroup,
        )
        mocker.patch(
            "oobleck.execution.engine.ReconfigurationEngine._copy_model_states",
            return_value=None,
        )
        mocker.patch(
            "oobleck.execution.engine.DataParallelEngine",
            return_value=MagicMock(),
        )
        yield FakeEngine(self, sample_args)

    @pytest.mark.parametrize(
        ["failed_ranks", "expected_ranks", "message"],
        [
            (
                [2],
                [[0, 1], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]],
                "reconfiguration base 1",
            ),
            (
                [6, 8],
                [[0, 1], [5, 7], [2, 3, 4], [9, 10, 11, 12, 13]],
                "reconfiguration base 2",
            ),
            (
                [10, 11],
                [[0, 1], [2, 3, 4], [9, 12, 13], [5, 6, 7, 8]],
                "reconfiguration base 3",
            ),
            # Pipeline 1 lacks a GPU, borrowing one from the last pipeline
            (
                [1],
                [[0, 13], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                "reconfiguration with borrowing GPUs 1",
            ),
            # Pipeline 1 and 2 lacks GPUs, borrowing them from the last pipeline
            (
                [1, 3, 4],
                [[0, 13], [2, 12], [9, 10, 11], [5, 6, 7, 8]],
                "reconfiguration with borrowing GPUs 2",
            ),
            # Pipeline 2 and 3 lacks GPUs, borrowing them from the last pipeline.
            (
                [2, 4, 6, 7, 8],
                [[0, 1], [3, 13], [5, 12], [9, 10, 11]],
                "reconfiguration with borrowing GPUs 3",
            ),
            # Merge pipeline 1 and 4
            (
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [[0, 12, 13]],
                "reconfiguration with pipeline merge 1",
            ),
            # Merge pipeline 1 and 2, and 3 and 4
            (
                [1, 2, 3, 5, 6, 7, 9, 11, 12, 13],
                [[0, 4], [8, 10]],
                "reconfiguration with pipeline merge 2",
            ),
            # Merge pipeline 1 and 2, and 3 and 4.
            # Pipeline 4 had enough nodes, but 3 doesn't so they should be merged.
            (
                [1, 2, 3, 5, 6, 7, 9, 10, 11],
                [[0, 4], [8, 12, 13]],
                "reconfiguration with pipeline merge 3",
            ),
        ],
        ids=[
            "base1",
            "base2",
            "base3",
            "borrow1",
            "borrow2",
            "borrow3",
            "pipeline_merge1",
            "pipeline_merge2",
            "pipeline_merge3",
        ],
    )
    def test_no_fsdp_reconfiguration(
        self,
        fake_engine: FakeEngine,
        failed_ranks: list[int],
        expected_ranks: list[list[int]],
        message: str,
    ):
        """
        FSDP disabled pipeline rank list before failures

        4 pipelines, 14 nodes, 14 GPUs total
        Pipeline 1 (2 nodes): [0, 1]
        Pipeline 2 (3 nodes): [2, 3,  4]
        Pipeline 3 (4 nodes): [5, 6,  7,  8]
        Pipeline 4 (5 nodes): [9, 10, 11, 12, 13]
        """
        logging.info(f"test_no_fsdp: {message}")

        pipelines = fake_engine.init_pipelines(num_gpus_per_node=1)
        assert len(pipelines) == 4
        # 2 + 3 + 4 + 5
        assert sum(len(pipeline._ranks) for pipeline in pipelines) == 14
        reconfigure_engine = ReconfigurationEngine(fake_engine, pipelines)

        # [0,1],[2,3,4],[5,6,7,8],[9,10,11,12,13]
        reconfigure_engine.on_reconfigure(failed_ranks)

        assert len(reconfigure_engine._pipelines) == len(expected_ranks)
        for pipeline, expected_rank in zip(
            reconfigure_engine._pipelines, expected_ranks
        ):
            assert pipeline._ranks == expected_rank

    @pytest.mark.parametrize(
        ["num_gpus_fsdp", "failed_ranks", "expected_ranks", "message"],
        [
            (
                2,
                [6, 7],
                [
                    list(range(0, 4)),
                    [4, 5, 8, 9],
                    list(range(10, 18)),
                    list(range(18, 28)),
                ],
                "FSDP2: reconfiguration base 1",
            ),
            (
                2,
                [10, 11, 18, 19],
                [
                    list(range(0, 4)),
                    list(range(4, 10)),
                    [12, 13, 14, 15, 16, 17],
                    [20, 21, 22, 23, 24, 25, 26, 27],
                ],
                "FSDP2: reconfiguration base 2",
            ),
            (
                4,
                [8, 9, 10, 11],
                [
                    list(range(0, 8)),
                    [12, 13, 14, 15, 16, 17, 18, 19],
                    list(range(20, 36)),
                    list(range(36, 56)),
                ],
                "FSDP4: reconfiguration base 1",
            ),
            (
                4,
                [20, 21, 22, 23, 28, 29, 30, 31],
                [
                    list(range(0, 8)),
                    [24, 25, 26, 27, 32, 33, 34, 35],
                    list(range(8, 20)),
                    list(range(36, 56)),
                ],
                "FSDP4: reconfiguration base 2",
            ),
            (
                2,
                [2, 3],
                [
                    [0, 1, 26, 27],
                    list(range(4, 10)),
                    list(range(10, 18)),
                    [18, 19, 20, 21, 22, 23, 24, 25],
                ],
                "FSDP2: reconfiguration with borrowing GPUs 1",
            ),
            (
                2,
                [2, 3, 4, 5, 8, 9],
                [
                    [0, 1, 26, 27],
                    [6, 7, 24, 25],
                    [18, 19, 20, 21, 22, 23],
                    list(range(10, 18)),
                ],
                "FSDP2: reconfiguration with borrowing GPUs 2",
            ),
            (
                2,
                [2, 3, 10, 11, 14, 15, 16, 17],
                [
                    [0, 1, 26, 27],
                    [12, 13, 24, 25],
                    list(range(4, 10)),
                    [18, 19, 20, 21, 22, 23],
                ],
                "FSDP2: reconfiguration with borrowing GPUs 3",
            ),
            (
                4,
                [4, 5, 6, 7],
                [
                    [0, 1, 2, 3, 52, 53, 54, 55],
                    list(range(8, 20)),
                    list(range(20, 36)),
                    [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                ],
                "FSDP4: reconfiguration with borrowing GPUs 1",
            ),
            (
                4,
                [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19],
                [
                    [0, 1, 2, 3, 52, 53, 54, 55],
                    [12, 13, 14, 15, 48, 49, 50, 51],
                    [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                    list(range(20, 36)),
                ],
                "FSDP4: reconfiguration with borrowing GPUs 2",
            ),
            (
                4,
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 36, 37, 38, 39],
                [
                    [0, 1, 2, 3, 52, 53, 54, 55],
                    [16, 17, 18, 19, 32, 33, 34, 35],
                    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                ],
                "FSDP2: reconfiguration with borrowing GPUs 3",
            ),
            (
                2,
                [2, 3, 8, 9, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27],
                [[10, 11, 12, 13], [18, 19, 20, 21], [0, 1, 4, 5, 6, 7]],
                "FSDP2: reconfiguration with pipeline merge 1",
            ),
            (
                2,
                [2, 3, 8, 9, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27],
                [[10, 11, 12, 13], [18, 19, 20, 21], [0, 1, 4, 5, 6, 7]],
                "FSDP2: reconfiguration with pipeline merge 2",
            ),
            (
                2,
                [2, 3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27],
                [[0, 1, 4, 5], [12, 13, 18, 19, 20, 21]],
                "FSDP2: reconfiguration with pipeline merge 3",
            ),
        ],
        ids=[
            "fsdp2_base1",
            "fsdp2_base2",
            "fsdp4_base1",
            "fsdp4_base2",
            "fsdp2_borrow1",
            "fsdp2_borrow2",
            "fsdp2_borrow3",
            "fsdp4_borrow1",
            "fsdp4_borrow2",
            "fsdp4_borrow3",
            "fsdp2_merge1",
            "fsdp2_merge2",
            "fsdp2_merge3",
        ],
    )
    def test_fsdp_reconfiguration(
        self,
        fake_engine: FakeEngine,
        num_gpus_fsdp: int,
        failed_ranks: list[int],
        expected_ranks: list[list[int]],
        message: str,
    ):
        """
        FSDP enabled pipeline rank list before failures

        2 GPUs per FSDP: 4 pipelines, 14 nodes, 28 GPUs total
        Pipeline 1 (2 nodes): [0,  1,  2,  3]
        Pipeline 2 (3 nodes): [4,  5,  6,  7,  8,  9]
        Pipeline 3 (4 nodes): [10, 11, 12, 13, 14, 15, 16, 17]
        Pipeline 4 (5 nodes): [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

        4 GPUs per FSDP: 4 pipelines, 14 nodes, 56 GPUs total
        Pipeline 1 (2 nodes): [0,  1,  2,  3,  4,  5,  6,  7]
        Pipeline 2 (3 nodes): [8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        Pipeline 3 (4 nodes): [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        Pipeline 4 (5 nodes): [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

        FSDP groups GPUs within the same node only,
        so node failure removes all FSDP groups at once.

        Unaffected expected ranks are represented as `list(range(start, end))`.
        """
        logging.info(f"test_fsdp: {message}")

        pipelines = fake_engine.init_pipelines(num_gpus_per_node=num_gpus_fsdp)
        assert len(pipelines) == 4

        num_gpus_in_total = 0
        for pipeline in pipelines:
            assert num_gpus_fsdp == pipeline._template._num_gpus_per_node
            num_gpus_in_total += len(pipeline._ranks)
        assert num_gpus_in_total == 14 * num_gpus_fsdp

        reconfigure_engine = ReconfigurationEngine(fake_engine, pipelines)

        reconfigure_engine.on_reconfigure(failed_ranks)
        assert len(reconfigure_engine._pipelines) == len(expected_ranks)
        for pipeline, expected_rank in zip(
            reconfigure_engine._pipelines, expected_ranks
        ):
            assert num_gpus_fsdp == pipeline._template._num_gpus_per_node
            assert pipeline._ranks == expected_rank
