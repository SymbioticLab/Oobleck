from __future__ import annotations

import asyncio
import copy
import multiprocessing
import re
import socket
import threading
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from unittest.mock import patch

import deepspeed.comm as dist
import pytest
import torch._C._distributed_c10d as c10d
import torch.distributed
from pytest_mock import MockerFixture
from torch.distributed.fsdp.flat_param import HandleShardingStrategy

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
)
from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataloader import OobleckSampler
from oobleck.execution.engine import OobleckEngine
from oobleck.execution.pipeline import OobleckPipeline
from tests.conftest import (
    TRAIN_BATCH_SIZE,
    OobleckMultiProcessTestCase,
    OobleckSingleProcessTestCase,
    OobleckStaticClassFactory,
)
from tests.elastic.conftest import OobleckElasticTestCase


class TestOobleckNumNodeComputationClass(OobleckSingleProcessTestCase):
    """
    Number of nodes calculation test
    Calculating minimum required number of nodes is based on
    profile results and device memory capacity.
    """

    factory: OobleckStaticClassFactory

    def get_fake_profile_results(self, num_layers: int) -> LayerExecutionResults:
        results: list[LayerExecutionResult] = []
        for index in range(num_layers):
            results.append(
                LayerExecutionResult(
                    layer_index=index,
                    forward=0.1,
                    backward=0.1,
                    allreduce_in_node={i + 1: 0.1 for i in range(8)},
                    allreduce_across_nodes={i + 1: 0.1 for i in range(64)},
                    mem_required=(1024, 0),
                )
            )
        return LayerExecutionResults(results)

    @pytest.fixture(scope="class")
    def pipe(self) -> tuple[connection.Connection, connection.Connection]:
        p1: connection.Connection
        p2: connection.Connection
        p1, p2 = multiprocessing.Pipe()
        yield p1, p2
        p1.close()
        p2.close()

    @dataclass
    class FakeDeviceProperties:
        total_memory: int

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        class_mocker: MockerFixture,
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> None:
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

        class_mocker.patch("torch.cuda.device_count", return_value=1)

        # This class does not mock get_pipeline_templates

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    @pytest.mark.parametrize(
        [
            "num_nodes",
            "num_gpus_per_node",
            "gpu_mem",
            "num_layers",
            "expected_min_num_nodes",
            "expect_fail",
        ],
        [
            (1, 1, 1024 * 32 * 6, 32, 1, False),
            (1, 1, 1024 * 32 * 6, 64, 2, True),
            (1, 1, 1024 * 128 * 6, 64, 1, False),
            (4, 1, 1024 * 32 * 6, 64, 2, False),
            (4, 1, 1024 * 16 * 6, 64, 4, False),
            (4, 1, 1024 * 16 * 6, 128, 8, True),
            (1, 4, 1024 * 16 * 6, 128, 2, True),
            (1, 4, 1024 * 32 * 6, 128, 1, False),
            (4, 4, 1024 * 16 * 6, 128, 2, False),
            (4, 4, 1024 * 1 * 6, 32, 8, True),
        ],
    )
    def test_multi_nodes_template_configuration(
        self,
        pipe: tuple[connection.Connection, connection.Connection],
        sample_args: OobleckArguments,
        num_nodes: int,
        num_gpus_per_node: int,
        gpu_mem: int,
        num_layers: int,
        expected_min_num_nodes: int,
        expect_fail: bool,
        mocker: MockerFixture,
    ):
        fake_profile = self.get_fake_profile_results(num_layers)

        # Fake device memory capacity so that number of nodes match with our simulated ones
        mocker.patch(
            "torch.cuda.get_device_properties",
            return_value=TestOobleckNumNodeComputationClass.FakeDeviceProperties(
                gpu_mem
            ),
        )
        mocker.patch(
            "oobleck.execution.engine.get_profile_results",
            return_value=fake_profile,
        )
        pt_create_mock = mocker.patch(
            "oobleck.csrc.planning.pipeline_template.PipelineTemplateGenerator.create_pipeline_templates",
            return_value=None,
        )

        if expect_fail:
            with pytest.raises(AssertionError) as e:
                OobleckEngine(0, num_nodes, num_gpus_per_node, pipe[1], sample_args)

            assert e.value.args[0].startswith("Minimum required number of nodes")
            match = re.search(r"minimum required: (\d+),", e.value.args[0])
            assert int(match[1]) == expected_min_num_nodes
        else:
            OobleckEngine(0, num_nodes, num_gpus_per_node, pipe[1], sample_args)
            pt_create_mock.assert_called_once_with(
                fake_profile, (expected_min_num_nodes, num_nodes), num_gpus_per_node
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
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> None:
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

        engine = OobleckEngine(0, 1, 1, pipe[1], sample_args)
        yield engine

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            dist.cdb = None

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

    # Agent should re-broadcast the port
    @staticmethod
    def broadcast_rank0_port(
        pipes: list[tuple[connection.Connection, connection.Connection]]
    ):
        port: int = pipes[0][0].recv()
        for pipe, _ in pipes:
            pipe.send(port)

    # All target methods must have the following signature:
    # (factory, rank, *args)
    @staticmethod
    def _run_distributed_engine(
        factory: OobleckStaticClassFactory,
        rank: int,
        num_stages: int,
        num_nodes: int,
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

        engine = OobleckEngine(0, num_nodes, num_gpus_per_node, pipe, arguments)
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
            # DistributionInfo
            pipe.send(DistributionInfo(agent_ips, len(agent_ips)))

        thread = threading.Thread(target=self.broadcast_rank0_port, args=(pipes,))
        thread.start()

        self.run_in_parallel(
            len(agent_ips),
            self._run_distributed_engine,
            num_stages,
            len(agent_ips),
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
        num_nodes: int,
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

        engine = OobleckEngine(0, num_nodes, num_gpus_per_node, pipe, arguments)
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
            # DistributionInfo
            pipe.send(DistributionInfo(agent_ips, len(agent_ips)))

        thread = threading.Thread(target=self.broadcast_rank0_port, args=(pipes,))
        thread.start()

        self.run_in_parallel(
            len(agent_ips),
            self._run_data_parallel_allreduce,
            num_stages,
            len(agent_ips),
            num_gpus_per_node,
            [p[1] for p in pipes],
            agent_ips,
            sample_args,
        )
        thread.join()

    @staticmethod
    def _run_fsdp_allreduce(
        factory: OobleckStaticClassFactory,
        rank: int,
        num_stages: int,
        num_nodes: int,
        pipe: list[connection.Connection],
        my_ip: str,
        arguments: OobleckArguments,
    ):
        # Assume all GPUs are in one GPUs
        num_gpus_per_node = 4
        num_nodes_per_pipeline = 1
        pipe = pipe[rank]
        global_num_microbatch = (
            arguments.global_microbatch_size // arguments.microbatch_size
        )

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

        engine = OobleckEngine(rank, num_nodes, num_gpus_per_node, pipe, arguments)
        engine.initialize_distributed()
        engine.instantiate_pipelines(global_num_microbatch)

        # Check only one pipelines are instantiated
        assert len(engine._reconfiguration._pipelines) == 1

        # Check all 4 GPUs are used
        assert len(engine._pipeline._ranks) == 4

        # Check FSDP is used
        for layer in engine._pipeline.execution._layers:
            if num_stages == 4:
                assert (
                    layer._param_handle._sharding_strategy
                    == HandleShardingStrategy.NO_SHARD
                )
                assert layer._group_size == 1
            else:
                assert (
                    layer._param_handle._sharding_strategy
                    == HandleShardingStrategy.FULL_SHARD
                )
                assert layer._group_size == 4 // num_stages

        engine._train_step()
        torch.cuda.synchronize()
        print(f"Rank {rank} finished training step")

    def test_fsdp_engine_train(self, num_stages: int, sample_args: OobleckArguments):
        ctx = multiprocessing.get_context("spawn")
        agent_ip: str = "127.0.0.1"
        pipes = [ctx.Pipe(duplex=True) for _ in range(4)]
        for pipe, _ in pipes:
            # DistributionInfo
            pipe.send(DistributionInfo([agent_ip], 4))

        thread = threading.Thread(target=self.broadcast_rank0_port, args=(pipes,))
        thread.start()

        num_gpus_per_node = 4 // num_stages
        self.run_in_parallel(
            4,
            self._run_fsdp_allreduce,
            num_stages,
            4,
            [p[1] for p in pipes],
            agent_ip,
            sample_args,
        )
        thread.join()

    @staticmethod
    def _run_reconfiguration(
        factory: OobleckStaticClassFactory,
        rank: int,
        num_stages: int,
        num_nodes: int,
        num_gpus_per_node: int,
        pipes: list[connection.Connection],
        agent_ips: list[str],
        arguments: OobleckArguments,
    ):
        pipe = pipes[rank]
        global_num_microbatch = (
            arguments.global_microbatch_size // arguments.microbatch_size
        )

        my_ip = agent_ips[rank]
        socket_patcher = patch("socket.gethostbyname", return_value=my_ip)
        socket_patcher.start()

        if num_stages == 1:
            pt_patcher = patch(
                "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
                return_value=[
                    factory.get_dummy_pipeline_template(
                        num_stages=1,
                        num_gpus_per_node=num_gpus_per_node,
                        num_nodes=1,
                    )
                ],
            )
        elif num_stages == 2:
            pt_patcher = patch(
                "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
                return_value=[
                    factory.get_dummy_pipeline_template(
                        num_stages=1,
                        num_gpus_per_node=num_gpus_per_node,
                        num_nodes=1,
                    ),
                    factory.get_dummy_pipeline_template(
                        num_stages=2,
                        num_gpus_per_node=num_gpus_per_node,
                        num_nodes=2,
                    ),
                ],
            )
        else:
            pt_patcher = patch(
                "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
                return_value=[
                    factory.get_dummy_pipeline_template(
                        num_stages=4,
                        num_gpus_per_node=num_gpus_per_node,
                        num_nodes=4,
                    ),
                ],
            )
        pt_patcher.start()

        # The test manually calls reconfiguration code. Remove listener.
        reconfig_mock = patch(
            "oobleck.execution.engine.ReconfigurationEngine._reconfiguration_listener_fn",
            return_value=None,
        )
        reconfig_mock.start()

        engine = OobleckEngine(0, num_nodes, num_gpus_per_node, pipe, arguments)
        engine.initialize_distributed()
        engine.instantiate_pipelines(global_num_microbatch)

        assert engine._dp_engine
        assert engine._reconfiguration

        assert engine._dist_info.agent_ips == agent_ips
        assert engine._dist_info.world_size == 4
        assert torch.distributed.get_world_size() == 4

        if rank == 3:
            return

        if num_stages == 1:
            # No copy happens. Expect dist.broadcast call number 0.
            with patch.object(
                torch.distributed, "broadcast", wraps=torch.distributed.broadcast
            ) as broadcast_spy:
                engine._reconfiguration._on_receive_reconfiguration_notification()
                assert broadcast_spy.call_count == 0
        elif num_stages == 2:
            # We should have one 1-stage pipeline and one 2-stage pipeline.
            # Copy must happen. Check rank grids.
            engine._reconfiguration._on_receive_reconfiguration_notification()

            model = factory.get_model()
            for pipeline in engine._reconfiguration._pipelines:
                assert len(pipeline._template.get_stages()) in [1, 2]
                if len(pipeline._template.get_stages()) == 1:
                    for layer_index, ranks in pipeline.rank_grid.items():
                        # This test didn't use FSDP
                        assert len(ranks) == 1
                        assert ranks[0] == 2
                else:
                    for layer_index, ranks in pipeline.rank_grid.items():
                        # This test didn't use FSDP
                        assert len(ranks) == 1
                        assert ranks[0] == (
                            0 if layer_index < len(model.layers) // 2 else 1
                        )
            rank = dist.get_rank()
            for layer_id, ranks_per_layer in engine._pipeline.rank_grid.items():
                if rank in ranks_per_layer:
                    layer = next(
                        layer
                        for layer in engine._pipeline.execution._layers
                        if layer.layer_id == layer_id
                    )
                    assert layer._param_handle.flat_param is not None
                    assert layer._param_handle.world_size == len(ranks_per_layer)
                else:
                    with pytest.raises(StopIteration):
                        next(
                            layer
                            for layer in engine._pipeline.execution._layers
                            if layer.layer_id == layer_id
                        )
        else:
            # We only have one pipeline and lose one node.
            # Cannot recover from it
            with pytest.raises(RuntimeError):
                engine._reconfiguration._on_receive_reconfiguration_notification()

        assert engine._dist_info.agent_ips == agent_ips[:3]
        assert engine._dist_info.world_size == 3

    def test_distribued_engine_reconfiguration(
        self, num_stages: int, sample_args: OobleckArguments
    ):
        # adjust global microbatch size so that batch distribution
        # works after losing 1 node.
        sample_args = copy.deepcopy(sample_args)
        sample_args.global_microbatch_size = 24 * TRAIN_BATCH_SIZE

        num_gpus_per_node = 1
        ctx = multiprocessing.get_context("spawn")
        agent_ips: list[str] = ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]
        pipes = [ctx.Pipe(duplex=True) for _ in range(len(agent_ips))]

        def broadcast_rank0_port():
            for pipe, _ in pipes:
                # DistributionInfo
                pipe.send(DistributionInfo(agent_ips, len(agent_ips)))

            # Rebroadcast rank 0 port.
            self.broadcast_rank0_port(pipes)

            for pipe, _ in pipes:
                # Broadcast that we lost node 3.
                pipe.send(agent_ips[3])

            # Rebroadcast rank 0 port again.
            self.broadcast_rank0_port(pipes)

        thread = threading.Thread(target=broadcast_rank0_port)
        thread.start()

        self.run_in_parallel(
            len(agent_ips),
            self._run_reconfiguration,
            num_stages,
            len(agent_ips),
            num_gpus_per_node,
            [p[1] for p in pipes],
            agent_ips,
            sample_args,
        )
        thread.join()
