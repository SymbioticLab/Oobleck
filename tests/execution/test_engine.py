from __future__ import annotations

import asyncio
import itertools
import multiprocessing
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

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.dataloader import OobleckSampler
from oobleck.execution.engine import OobleckEngine, ReconfigurationEngine
from oobleck.execution.pipeline import OobleckPipeline
from tests.conftest import (
    TRAIN_BATCH_SIZE,
    OobleckElasticTestCase,
    OobleckMultiProcessTestCase,
    OobleckSingleProcessTestCase,
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
        global_microbatch_size=16 * TRAIN_BATCH_SIZE,
    )


@pytest.mark.skip(reason="asyncio hangs when running multiple tests")
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
        class_mocker.patch("socket.gethostname", return_value="127.0.0.1")

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


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test requires 4 GPUs")
@pytest.mark.parametrize("num_stages", [1, 2, 4], ids=["1stage", "2stages", "4stages"])
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

    # All target methods must have the following signature:
    # (factory, rank, *args)
    @staticmethod
    def _run_distributed_engine(
        factory: OobleckStaticClassFactory,
        rank: int,
        num_stages: int,
        num_gpus_per_node: int,
        pipe: list[connection.Connection],
        agent_ips: list[str],
        arguments: OobleckArguments,
    ):
        num_nodes_per_pipeline = num_stages
        pipe = pipe[rank]

        my_ip = agent_ips[rank]
        socket_patcher = patch("socket.gethostbyname", return_value=my_ip)
        socket_patcher.start()
        pt_patcher = patch(
            "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
            return_value=[
                factory.get_dummy_pipeline_template(
                    num_stages=num_stages,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=num_nodes_per_pipeline,
                )
            ],
        )
        pt_patcher.start()

        engine = OobleckEngine(pipe, arguments)
        engine.initialize_distributed()
        assert dist.get_rank() < dist.get_world_size()
        assert dist.get_world_size() == 4, "This test must run with 4 GPUs"
        global_num_microbatch = (
            arguments.global_microbatch_size // arguments.microbatch_size
        )
        engine.instantiate_pipelines(global_num_microbatch)

        # Check it uses expected pipeline template and pipeline
        expected = factory.get_dummy_pipeline_template(
            num_stages=num_stages,
            num_gpus_per_node=num_gpus_per_node,
            num_nodes=num_nodes_per_pipeline,
        )
        assert engine._pipeline_templates == [expected]
        assert engine._pipeline
        assert engine._pipeline._template == expected

        # OobleckSampler has a list of num_microbatches for all pipelines.
        # Sum of number of microbatches must be equal to global # microbatches
        world_size = dist.get_world_size()
        sampler: OobleckSampler = engine._pipeline._dataloader.batch_sampler
        assert len(sampler.num_microbatches) == world_size // (
            num_stages * num_gpus_per_node
        )
        assert sum(sampler.num_microbatches) == global_num_microbatch

    def test_distributed_engine(self, num_stages: int, sample_args: OobleckArguments):
        num_gpus_per_node = 1
        ctx = multiprocessing.get_context("spawn")
        agent_ips: list[str] = ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]
        pipes = [ctx.Pipe(duplex=True) for _ in range(len(agent_ips))]
        for pipe, _ in pipes:
            # max num GPUs
            pipe.send(num_gpus_per_node)
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
            num_stages,
            num_gpus_per_node,
            [p[1] for p in pipes],
            agent_ips,
            sample_args,
        )
        thread.join()

        [p[0].close() for p in pipes]
        [p[1].close() for p in pipes]

    @staticmethod
    def _run_data_parallel_allreduce(
        factory: OobleckStaticClassFactory,
        rank: int,
        num_stages: int,
        num_gpus_per_node: int,
        pipe: list[connection.Connection],
        agent_ips: list[str],
        arguments: OobleckArguments,
    ):
        num_nodes_per_pipeline = num_stages
        pipe = pipe[rank]
        global_num_microbatch = (
            arguments.global_microbatch_size // arguments.microbatch_size
        )

        my_ip = agent_ips[rank]
        socket_patcher = patch("socket.gethostbyname", return_value=my_ip)
        socket_patcher.start()
        pt_patcher = patch(
            "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
            return_value=[
                factory.get_dummy_pipeline_template(
                    num_stages=num_stages,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=num_nodes_per_pipeline,
                )
            ],
        )
        pt_patcher.start()

        engine = OobleckEngine(pipe, arguments)
        engine.initialize_distributed()
        engine.instantiate_pipelines(global_num_microbatch)

        # Monitor layer allreduce is called
        with patch.object(
            torch.distributed, "all_reduce", wraps=torch.distributed.all_reduce
        ) as allreduce_spy:
            engine._train_step()
            torch.cuda.synchronize()
            print(f"Rank {rank} finished training step")
            expected_call_number = len(engine._pipeline.execution._layers)
            assert allreduce_spy.call_count == expected_call_number, (
                f"torch.distributed.allreduce expected to be called {expected_call_number} times, "
                f"but called {allreduce_spy.call_count} times."
            )

        # Optimizer must have its own state
        p: torch.nn.Parameter
        optimizer = engine._pipeline.execution._optimizer
        for p in optimizer.param_groups[0]["params"]:
            if p.numel() == 0:
                continue
            assert all(
                key in optimizer.state[p] for key in ["step", "exp_avg", "exp_avg_sq"]
            )

    def test_distributed_engine_train(
        self, num_stages: int, sample_args: OobleckArguments
    ):
        num_gpus_per_node = 1
        ctx = multiprocessing.get_context("spawn")
        agent_ips: list[str] = ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]
        pipes = [ctx.Pipe(duplex=True) for _ in range(len(agent_ips))]
        for pipe, _ in pipes:
            # max num GPUs
            pipe.send(num_gpus_per_node)
            # DistributionInfo
            pipe.send(DistributionInfo(agent_ips, len(agent_ips)))

        def broadcast_rank0_port():
            port: int = pipes[0][0].recv()
            for pipe, _ in pipes:
                pipe.send(port)

        thread = threading.Thread(target=broadcast_rank0_port)
        thread.start()

        self.run_in_parallel(
            len(agent_ips),
            self._run_data_parallel_allreduce,
            num_stages,
            num_gpus_per_node,
            [p[1] for p in pipes],
            agent_ips,
            sample_args,
        )
        thread.join()


class TestOobleckReconfigurationClass(OobleckSingleProcessTestCase):
    class FakePipeline:
        def __init__(
            self,
            pipeline_id: int,
            pipeline_template: PipelineTemplate,
            ranks: list[int],
            *args,
            **kwargs,
        ):
            self._pipeline_id = pipeline_id
            self._template = pipeline_template
            self._ranks = ranks
            self._global_step = 0
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

    class FakeEngine:
        def __init__(
            self,
            test_class: TestOobleckReconfigurationClass,
            sample_args: OobleckArguments,
        ):
            self.test_class = test_class
            self._agent_pipe: connection.Connection = None
            self._args = sample_args
            self._hf_training_args = test_class.factory._training_args
            self._num_nodes = sum(range(2, 6))
            self._num_gpus_per_node = 1
            self._dataset = test_class.factory.get_dataset()
            self._model = test_class.factory.get_model()
            self._profile_results = test_class.factory.get_dummy_profile()
            self._pipeline_templates = [
                test_class.factory.get_dummy_pipeline_template(
                    num_stages=i,
                    num_gpus_per_node=1,
                    num_nodes=i,
                )
                for i in range(2, 6)
            ]

        def init_pipelines(self) -> list[OobleckPipeline]:
            num_gpus_used: int = 0
            pipeline_id: int = 0
            pipelines: list[OobleckPipeline] = []

            # Have one pipelines for each pipeline template
            for template in self._pipeline_templates:
                pipelines.append(
                    self.test_class.FakePipeline(
                        pipeline_id,
                        template,
                        list(
                            range(
                                num_gpus_used,
                                num_gpus_used + len(template.get_stages()),
                            )
                        ),
                    )
                )

                pipeline_id += 1
                num_gpus_used += len(template.get_stages())
            self._pipeline = pipelines[0]
            return pipelines

    @pytest.fixture
    def fake_engine(
        self, mocker: MockerFixture, sample_args: OobleckArguments
    ) -> TestOobleckReconfigurationClass.FakeEngine:
        mocker.patch(
            "oobleck.planning.instantiator.OobleckPipeline",
            new=TestOobleckReconfigurationClass.FakePipeline,
        )
        mocker.patch(
            "deepspeed.comm.new_group",
            return_value=None,
        )
        mocker.patch(
            "oobleck.execution.engine.ReconfigurationEngine._copy_model_states",
            return_value=None,
        )
        yield self.FakeEngine(self, sample_args)

    @pytest.mark.parametrize(
        ["failed_ranks", "expected_ranks"],
        [
            ([2], [[0, 1], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]]),
            ([6, 8], [[0, 1], [5, 7], [2, 3, 4], [9, 10, 11, 12, 13]]),
            ([10, 11], [[0, 1], [2, 3, 4], [9, 12, 13], [5, 6, 7, 8]]),
        ],
    )
    def test_reconfigure_with_all_available_templates(
        self,
        fake_engine: TestOobleckReconfigurationClass.FakeEngine,
        failed_ranks: list[int],
        expected_ranks: list[list[int]],
    ):
        pipelines = fake_engine.init_pipelines()
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
        ["failed_ranks", "expected_ranks"],
        [
            ([1], [[0, 13], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ([1, 3, 4], [[0, 13], [2, 12], [9, 10, 11], [5, 6, 7, 8]]),
            ([2, 4, 6, 7, 8], [[0, 1], [3, 13], [5, 12], [9, 10, 11]]),
        ],
    )
    def test_reconfigure_borrow_nodes(
        self,
        fake_engine: TestOobleckReconfigurationClass.FakeEngine,
        failed_ranks: list[int],
        expected_ranks: list[list[int]],
    ):
        pipelines = fake_engine.init_pipelines()
        assert len(pipelines) == 4
        assert sum(len(pipeline._ranks) for pipeline in pipelines) == 14
        reconfigure_engine = ReconfigurationEngine(fake_engine, pipelines)

        reconfigure_engine.on_reconfigure(failed_ranks)

        assert len(reconfigure_engine._pipelines) == len(expected_ranks)
        for pipeline, expected_rank in zip(
            reconfigure_engine._pipelines, expected_ranks
        ):
            assert pipeline._ranks == expected_rank

    @pytest.mark.parametrize(
        ["failed_ranks", "expected_ranks"],
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [[0, 12, 13]]),
            ([1, 2, 3, 5, 6, 7, 9, 11, 12, 13], [[0, 4], [8, 10]]),
            ([1, 2, 3, 4, 5, 6, 7, 8], [[0, 13], [9, 10, 11, 12]]),
            ([1, 2, 3, 5, 6, 7, 9, 10, 11], [[0, 4], [8, 12, 13]]),
        ],
    )
    def test_reconfigure_merge_pipelines(
        self,
        fake_engine: TestOobleckReconfigurationClass.FakeEngine,
        failed_ranks: list[int],
        expected_ranks: list[list[int]],
    ):
        pipelines = fake_engine.init_pipelines()
        assert len(pipelines) == 4
        assert sum(len(pipeline._ranks) for pipeline in pipelines) == 14
        reconfigure_engine = ReconfigurationEngine(fake_engine, pipelines)

        reconfigure_engine.on_reconfigure(failed_ranks)

        assert len(reconfigure_engine._pipelines) == len(expected_ranks)
        for pipeline, expected_rank in zip(
            reconfigure_engine._pipelines, expected_ranks
        ):
            assert pipeline._ranks == expected_rank
