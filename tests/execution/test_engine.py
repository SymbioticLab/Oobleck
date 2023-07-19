from __future__ import annotations

import asyncio
import multiprocessing
import os
import socket
import threading
import traceback
from multiprocessing import connection
from pathlib import Path
from unittest.mock import patch

import deepspeed.comm as dist
import pytest
import torch._C._distributed_c10d as c10d
import torch.distributed
from pytest_mock import MockerFixture

from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.engine import OobleckEngine
from oobleck.execution.pipeline import OobleckPipeline
from tests.conftest import (
    TRAIN_BATCH_SIZE,
    OobleckElasticTestCase,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
    datasets,
    model_args,
)


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
        global_microbatch_size=512,
    )


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
        # max num GPUs
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

    @pytest.mark.asyncio
    async def test_init_engine_pipeline(
        self,
        pipe: tuple[connection.Connection, connection.Connection],
        engine: OobleckEngine,
        event_loop: asyncio.AbstractEventLoop,
        sample_args: OobleckArguments,
        mocker: MockerFixture,
    ):
        pipe[0].send(DistributionInfo([socket.gethostbyname(socket.gethostname())], 1))

        def rebroadcast() -> int:
            port = pipe[0].recv()
            pipe[0].send(port)
            return port

        future = event_loop.run_in_executor(None, rebroadcast)
        engine.initialize_distributed()
        await asyncio.wait_for(future, timeout=5)

        init_pipeline_spy = mocker.spy(
            OobleckPipeline, "initialize_distributed_pipeline"
        )

        global_num_microbatch = (
            sample_args.global_microbatch_size // sample_args.microbatch_size
        )
        engine.instantiate_pipelines(global_num_microbatch)

        expected_pipeline_template = self.factory.get_dummy_pipeline_template(
            num_stages=1,
            num_gpus_per_node=1,
            num_nodes=1,
        )
        assert engine._num_nodes == 1
        assert engine._pipeline
        assert engine._pipeline._template == expected_pipeline_template
        assert init_pipeline_spy.call_count == 1


class TestOobleckDistributedEngineClass(OobleckMultiProcessTestCase):
    @staticmethod
    def _worker_init(
        queue: multiprocessing.Queue,
        rank: int,
        world_size: int,
        model_name: str,
        directory: Path,
        test: callable,
        *args,
    ):
        """
        OobleckEngine initializes distributed inside it,
        so we need to avoid automatic distributed env initialization.
        """
        try:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(rank))
            monkeypatch.delenv("RANK", raising=False)
            monkeypatch.delenv("WORLD_SIZE", raising=False)
            monkeypatch.delenv("MASTER_ADDR", raising=False)
            monkeypatch.delenv("MASTER_PORT", raising=False)

            patcher = patch("torch.cuda.device_count", return_value=1)
            patcher.start()

            factory = OobleckStaticClassFactory(model_name, directory)
            torch.cuda.set_device(0)

            with patch(
                "oobleck.execution.engine.OobleckModel",
                return_value=factory.get_model(),
            ), patch(
                "oobleck.execution.engine.get_profile_results",
                return_value=factory.get_dummy_profile(),
            ), patch(
                "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
                return_value=[
                    factory.get_dummy_pipeline_template(
                        num_stages=1, num_gpus_per_node=1, num_nodes=1
                    )
                ],
            ):
                result = test(factory, rank, *args)

            queue.put(
                {
                    "success": (result if result is not None else ""),
                    "rank": rank,
                }
            )
        except Exception as e:
            queue.put({"error": str(e) + "\n" + traceback.format_exc()})

    @staticmethod
    def _run_distributed_engine(
        factory: OobleckStaticClassFactory,
        rank: int,
        pipe: list[connection.Connection],
        agent_ips: list[str],
        arguments: OobleckArguments,
    ):
        pipe = pipe[rank]

        my_ip = agent_ips[rank]
        patcher = patch("socket.gethostbyname", return_value=my_ip)
        patcher.start()

        engine = OobleckEngine(pipe, arguments)
        engine.initialize_distributed()
        assert dist.get_rank() < dist.get_world_size()
        global_num_microbatch = (
            arguments.global_microbatch_size // arguments.microbatch_size
        )
        engine.instantiate_pipelines(global_num_microbatch)

        # Check it uses expected pipeline template and pipeline
        expected = factory.get_dummy_pipeline_template(
            num_stages=1, num_gpus_per_node=1, num_nodes=1
        )
        assert engine._pipeline_templates == [expected]
        assert engine._pipeline
        assert engine._pipeline._template == expected

        # OobleckSampler has a list of num_microbatches for all pipelines.
        # Sum of number of microbatches must be equal to global # microbatches
        world_size = dist.get_world_size()
        assert len(engine._dataloader.batch_sampler.num_microbatches) == world_size
        assert (
            sum(engine._dataloader.batch_sampler.num_microbatches)
            == global_num_microbatch
        )

    def test_distributed_engine(self, sample_args: OobleckArguments):
        ctx = multiprocessing.get_context("spawn")
        agent_ips: list[str] = ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]
        pipes = [ctx.Pipe(duplex=True) for _ in range(len(agent_ips))]
        for pipe, _ in pipes:
            # max num GPUs
            pipe.send(1)
            # DistributionInfo
            pipe.send(DistributionInfo(agent_ips, len(agent_ips)))

        # Agent should re-broadcast the port
        def broadcast_rank0_port():
            port: int = pipes[0][0].recv()
            for pipe, _ in pipes:
                pipe.send(port)

        thread = threading.Thread(target=broadcast_rank0_port)
        thread.start()

        self.run_in_parallel(
            len(agent_ips),
            self._run_distributed_engine,
            [p[1] for p in pipes],
            agent_ips,
            sample_args,
        )
        thread.join()

        [p[0].close() for p in pipes]
        [p[1].close() for p in pipes]
