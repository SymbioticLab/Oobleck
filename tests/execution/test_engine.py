from __future__ import annotations

import asyncio
import multiprocessing
import socket
from multiprocessing import connection

import deepspeed.comm as dist
import pytest
import torch._C._distributed_c10d as c10d
import torch.distributed
from pytest_mock import MockerFixture

from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.engine import OobleckEngine
from tests.conftest import (
    OobleckElasticTestCase,
    OobleckStaticClassFactory,
    datasets,
    model_args,
)


@pytest.mark.skip
class TestOobleckEngineClass(OobleckElasticTestCase):
    factory: OobleckStaticClassFactory

    @pytest.fixture(scope="class")
    def pipe(self) -> tuple[connection.Connection, connection.Connection]:
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
        pipe: tuple[connection.Connection, connection.Connection],
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
                cls.factory.get_dummy_pipeline_template(
                    num_stages=num_gpus + 1,
                    num_gpus_per_node=num_gpus + 1,
                    num_nodes=1,
                )
                for num_gpus in range(4)
            ],
        )

        yield

    @pytest.fixture
    def engine(
        self,
        pipe: tuple[connection.Connection, connection.Connection],
        sample_args: OobleckArguments,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> OobleckEngine:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        mocker.patch("torch.cuda.device_count", return_value=1)
        engine = OobleckEngine(pipe[1], sample_args)
        yield engine

    def test_init_engine(self, engine: OobleckEngine):
        assert not torch.distributed.is_initialized()
        assert len(engine._pipeline_templates) == 4

    @pytest.mark.asyncio
    async def test_init_engine_with_elastic(
        self,
        pipe: tuple[connection.Connection, connection.Connection],
        engine: OobleckEngine,
        event_loop: asyncio.AbstractEventLoop,
        mocker: MockerFixture,
    ):
        # Spy torch.distributed
        torch_init_spy = mocker.spy(torch.distributed, "init_process_group")

        # An agent is supposed to send DistributionInfo
        pipe[0].send(DistributionInfo([socket.gethostbyname(socket.gethostname())], 1))

        # An engine is supposed to bind a port and send it,
        # and an agent must re-broadcast it.
        def rebroadcast() -> int:
            port = pipe[0].recv()
            pipe[0].send(port)
            return port

        future = event_loop.run_in_executor(None, rebroadcast)
        engine.initialize_distributed()

        await asyncio.wait_for(future, timeout=5)

        port: int = future.result()
        store: torch.distributed.distributed_c10d.PrefixStore = (
            torch.distributed.distributed_c10d._get_default_store()
        )
        assert isinstance(store.underlying_store, c10d.TCPStore)
        assert store.underlying_store.port == port

        assert torch_init_spy.call_count == 1
        assert torch.distributed.is_initialized()
        assert dist.is_initialized()
