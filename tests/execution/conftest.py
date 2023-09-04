import pytest

from oobleck.elastic.training_util import (
    DistributedArguments,
    JobArguments,
    ModelArguments,
    OobleckArguments,
)
from tests.conftest import TRAIN_BATCH_SIZE, datasets, model_args


@pytest.fixture(scope="module")
def sample_args(model_name_fixture: str) -> OobleckArguments:
    dataset: tuple[str, (str | None)] = datasets[model_name_fixture]

    return OobleckArguments(
        dist=DistributedArguments(
            master_ip="127.0.0.1",
            master_port=0,
            node_ips=["127.0.0.1"],
        ),
        job=JobArguments(
            fault_threshold=1,
            microbatch_size=TRAIN_BATCH_SIZE,
            global_microbatch_size=4 * TRAIN_BATCH_SIZE,
        ),
        model=ModelArguments(
            model_name=model_name_fixture,
            model_tag="test",
            dataset_path=dataset[0],
            dataset_name=dataset[1],
            model_args=model_args[model_name_fixture],
        ),
    )
