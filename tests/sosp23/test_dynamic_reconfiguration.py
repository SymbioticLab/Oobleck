import logging

import pytest

from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.engine import ReconfigurationEngine
from tests.sosp23.conftest import EvalFakeEngine


@pytest.mark.parametrize(
    "failed_ranks",
    [
        [2],
        [1, 3, 4],
        [2, 4, 6, 7, 8],
        [1, 2, 3, 5, 6, 7, 9, 10, 11],
    ],
)
def test_reinstantiate_pipeline_no_pipeline_merge(
    logger: logging.Logger,
    fake_engine: EvalFakeEngine,
    sample_args: OobleckArguments,
    failed_ranks: list[int],
):
    # Initial rank distribution:
    # [0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]

    pipelines = fake_engine.init_pipelines()
    logger.info(
        f"""PIPELINE RECONFIGURATION
        Number of pipelines: {len(pipelines)}
        Number of ranks used by each pipelines: {[len(pipeline._ranks) for pipeline in pipelines]} 
        Pipeline rank distribution: {[pipeline._ranks for pipeline in pipelines]}
        Reconfiguration start after losing {failed_ranks} ranks
        ==============================
        """
    )
    reconfigure_engine = ReconfigurationEngine(fake_engine, pipelines)
    reconfigure_engine.on_reconfigure(failed_ranks)

    if failed_ranks == [2]:
        logger.info(
            """PIPELINE CONDITION AFTER FAILURES
            Pipeline 1: [0, 1]              (no change)
            Pipeline 2: [3, 4]              (lost one rank. need reconfiguration)
            Pipeline 3: [5, 6, 7, 8]        (no change)
            Pipeline 4: [9, 10, 11, 12, 13] (no change)
            ==============================
            """
        )
        expected_ranks = [[0, 1], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]]
        logger.info(
            f"""AFTER PIPELINE RECONFIGURATION (case 1)
            - Pipeline 2: reinstantiate 2-stage pipeline template
            Expected rank distribution: {expected_ranks}
            Rank distribution         : {[pipeline._ranks for pipeline in reconfigure_engine._pipelines]}
            ==============================
            """
        )
    elif failed_ranks == [1, 3, 4]:
        logger.info(
            """PIPELINE CONDITION AFTER FAILURES
            Pipeline 1: [0]                 (lost one rank. no feasible pipeline template. need to borrow rank)
            Pipeline 2: [2]                 (lost two ranks. no feasible pipeline template. need to borrow rank)
            Pipeline 3: [5, 6, 7, 8]        (no change)
            Pipeline 4: [9, 10, 11, 12, 13] (no change)
            ==============================
            """
        )
        expected_ranks = [[0, 13], [2, 12], [9, 10, 11], [5, 6, 7, 8]]
        logger.info(
            f"""EXPECTED RECONFIGURED PIPELINES (case 2)
            - [borrow rank 13] 4th pipeline -> 1st pipeline
            - [borrow rank 12] 4th pipeline -> 2nd pipeline
            - Pipeline 1: reinstantiate 2-stage pipeline template
            - Pipeline 2: reinstantiate 2-stage pipeline template
            - Pipeline 3: reinstantiate 3-stage pipeline template
            Expected rank distribution: {expected_ranks}
            Rank distribution         : {[pipeline._ranks for pipeline in reconfigure_engine._pipelines]}
            ==============================
            """
        )
    elif failed_ranks == [2, 4, 6, 7, 8]:
        logger.info(
            """PIPELINE CONDITION AFTER FAILURES
            Pipeline 1: [0, 1]              (no change)
            Pipeline 2: [3]                 (lost two ranks. no feasible pipeline template. need to borrow rank)
            Pipeline 3: [5]                 (lost three ranks. no feasible pipeline template. need to borrow rank)
            Pipeline 4: [9, 10, 11, 12, 13] (no change)
            ==============================
            """
        )
        expected_ranks = [[0, 1], [3, 13], [5, 12], [9, 10, 11]]
        logger.info(
            f"""EXPECTED RECONFIGURED PIPELINES (case 2)
            - [borrow rank 13] 4th pipeline -> 3rd pipeline
            - [borrow rank 12] 4th pipeline -> 2nd pipeline
            - Pipeline 2: reinstantiate 2-stage pipeline template
            - Pipeline 3: reinstantiate 2-stage pipeline template
            - Pipeline 4: reinstantiate 3-stage pipeline template
            Expected rank distribution: {expected_ranks}
            Rank distribution         : {[pipeline._ranks for pipeline in reconfigure_engine._pipelines]}
            ==============================
            """
        )
    else:
        logger.info(
            """PIPELINE CONDITION AFTER FAILURES
            Pipeline 1: [0]                 (lost one rank. no feasible pipeline template. need to borrow rank)
            Pipeline 2: [4]                 (lost two ranks. no feasible pipeline template. need to borrow rank)
            Pipeline 3: [8]                 (lost all ranks. Remove it)
            Pipeline 4: [12, 13]            (lost three ranks. need reconfiguration)
            ==============================
            """
        )
        expected_ranks = [[0, 4], [8, 12, 13]]
        logger.info(
            f"""EXPECTED RECONFIGURED PIPELINES (case 3)
            - [no borrow available] merge pipeline 1 and 2 -> [0, 4]: reinstantiate 2-stage pipeline template
            - [no borrow available] merge pipeline 3 and 4 -> [8, 12, 13]: reinstantiate 3-stage pipeline template
            Expected rank distribution: {expected_ranks}
            Rank distribution         : {[pipeline._ranks for pipeline in reconfigure_engine._pipelines]}
            ==============================
            """
        )

    assert expected_ranks == [
        pipeline._ranks for pipeline in reconfigure_engine._pipelines
    ]
