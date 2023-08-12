import pytest

from oobleck.elastic.training_util import OobleckArguments
from tests.conftest import TRAIN_BATCH_SIZE, datasets, model_args


@pytest.fixture(scope="module")
def sample_args(model_name_fixture: str) -> OobleckArguments:
    dataset: tuple[str, (str | None)] = datasets[model_name_fixture]
    return OobleckArguments(
        model_name=model_name_fixture,
        model_tag="test",
        dataset_path=dataset[0],
        dataset_name=dataset[1],
        fault_threshold=1,
        model_args=model_args[model_name_fixture],
        microbatch_size=TRAIN_BATCH_SIZE,
        global_microbatch_size=4 * TRAIN_BATCH_SIZE,
    )
