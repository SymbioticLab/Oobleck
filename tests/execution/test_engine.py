from __future__ import annotations

import multiprocessing
from multiprocessing import connection
from unittest.mock import MagicMock

import pytest
import torch.distributed

import oobleck.execution.engine as engine_module
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
            model_args=model_args[model_name_fixture],
        )

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        pipe: tuple(connection.Connection, connection.Connection),
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> None:
        pipe[0].send(4)

        directory = tmp_path_factory.getbasetemp()
        request.cls.factory = OobleckStaticClassFactory(model_name_fixture, directory)
        engine_module.OobleckDataset = MagicMock(return_value=cls.factory.get_dataset())
        engine_module.OobleckModel = MagicMock(return_value=cls.factory.get_model())
        engine_module.get_profile_results = MagicMock(
            return_value=cls.factory.get_dummy_profile()
        )
        engine_module.PipelineTemplateGenerator.create_pipeline_templates = MagicMock(
            spec=["layer_execution_results", "num_nodes", "num_gpus_per_node"],
            return_value=[
                cls.factory.get_dummy_pipeline_template(num_gpus + 1)
                for num_gpus in range(4)
            ],
        )

    def test_init_engine(
        self,
        pipe: tuple(connection.Connection, connection.Connection),
        sample_args: OobleckArguments,
    ):
        engine = OobleckEngine(pipe[1], sample_args)
        assert not torch.distributed.is_initialized()
