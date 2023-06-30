from __future__ import annotations

import multiprocessing
from multiprocessing import connection

import pytest
import torch.distributed
from pytest_mock import MockerFixture

from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.engine import OobleckEngine
from tests.conftest import (
    OobleckElasticTestCase,
    OobleckStaticClassFactory,
    datasets,
    model_args,
)


class TestOobleckEngineClass(OobleckElasticTestCase):
    factory: OobleckStaticClassFactory

    @pytest.fixture(scope="class")
    def pipe(self) -> tuple(connection.Connection, connection.Connection):
        p1: connection.Connection
        p2: connection.Connection
        p1, p2 = multiprocessing.Pipe()
        yield p1, p2
        p1.close()
        p2.close()

    @pytest.fixture(scope="class")
    def sample_args(self, model_name_fixture: str) -> OobleckArguments:
        dataset: tuple[str, (str | None)] = datasets[model_name_fixture]
        return OobleckArguments(
            model_name=model_name_fixture,
            model_tag="test",
            dataset_path=dataset[0],
            dataset_name=dataset[1],
        )

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        class_mocker: MockerFixture,
        pipe: tuple(connection.Connection, connection.Connection),
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> None:
        pipe[0].send(4)

        directory = tmp_path_factory.getbasetemp()
        request.cls.factory = OobleckStaticClassFactory(model_name_fixture, directory)

        class_mocker.patch(
            "oobleck.execution.engine.OobleckDataset",
            return_value=cls.factory.get_dataset(),
        )
        class_mocker.patch(
            "oobleck.execution.engine.OobleckModel",
            return_value=cls.factory.get_model(),
        )
        class_mocker.patch(
            "oobleck.execution.engine.get_profile_results",
            return_value=cls.factory.get_dummy_profile(),
        )
        class_mocker.patch(
            "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
            return_value=[
                cls.factory.get_dummy_pipeline_template(num_gpus + 1)
                for num_gpus in range(4)
            ],
        )

        yield

    def test_init_engine(
        self,
        pipe: tuple(connection.Connection, connection.Connection),
        sample_args: OobleckArguments,
    ):
        engine = OobleckEngine(pipe[1], sample_args)
        assert not torch.distributed.is_initialized()
        assert len(engine._pipeline_templates) == 4
