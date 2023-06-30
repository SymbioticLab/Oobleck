from __future__ import annotations

import multiprocessing
from multiprocessing import connection
from unittest.mock import MagicMock, patch

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
    @patch("engine_module.OobleckDataset")
    @patch("engine_module.OobleckModel")
    @patch("engine_module.get_profile_results")
    @patch("engine_module.PipelineTemplateGenerator.create_pipeline_templates")
    def setup_class(
        cls,
        mock_oobleck_dataset: MagicMock,
        mock_oobleck_model: MagicMock,
        mock_get_profile_results: MagicMock,
        mock_create_pipeline_templates: MagicMock,
        pipe: tuple(connection.Connection, connection.Connection),
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> None:
        pipe[0].send(4)

        directory = tmp_path_factory.getbasetemp()
        request.cls.factory = OobleckStaticClassFactory(model_name_fixture, directory)

        mock_oobleck_dataset.return_value = cls.factory.get_dataset()
        mock_oobleck_model.return_value = cls.factory.get_model()
        mock_get_profile_results.return_value = cls.factory.get_dummy_profile()
        mock_create_pipeline_templates.return_value = [
            cls.factory.get_dummy_pipeline_template(num_gpus + 1)
            for num_gpus in range(4)
        ]

        yield

    def test_init_engine(
        self,
        pipe: tuple(connection.Connection, connection.Connection),
        sample_args: OobleckArguments,
    ):
        engine = OobleckEngine(pipe[1], sample_args)
        assert not torch.distributed.is_initialized()
        assert len(engine._pipeline_templates) == 4
