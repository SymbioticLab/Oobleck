import asyncio
import functools
import signal
from concurrent.futures import Future, ThreadPoolExecutor, wait
from multiprocessing import connection
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pytest_mock import MockerFixture

import oobleck.elastic.message_util as message_util
from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import OobleckMasterDaemon, _AgentInfo
from oobleck.elastic.training_util import OobleckArguments
from oobleck.elastic.worker import worker_main
from tests.conftest import OobleckStaticClassFactory, datasets, model_args
from tests.elastic.conftest import OobleckElasticTestCase


class TestOobleckAgentClassWithNoDaemon:
    pass


class TestOobleckAgentClass(OobleckElasticTestCase):
    @pytest.mark.asyncio
    async def test_register_agent(self, agent: OobleckAgent):
        agent.send_request = AsyncMock(wraps=agent.send_request)
        await agent.register_agent()

        await asyncio.sleep(1)
        agent.send_request.assert_called_with(message_util.RequestType.PING, None, None)

    @pytest.mark.asyncio
    async def test_get_dist_info(self, agent: OobleckAgent):
        await agent.register_agent()

        agent.send_request = AsyncMock(wraps=agent.send_request)
        agent.on_receive_dist_info = AsyncMock(wraps=agent.on_receive_dist_info)
        await agent.get_dist_info()

        await asyncio.sleep(0.2)
        agent.send_request.assert_called_with(
            message_util.RequestType.GET_DIST_INFO, None, agent.on_receive_dist_info
        )
        agent.on_receive_dist_info.assert_called()

    @pytest.mark.asyncio
    async def test_receive_reconfiguration(
        self, daemon: OobleckMasterDaemon, agent: OobleckAgent
    ):
        await agent.register_agent()
        # TODO: daemon only check IP when registering agent, then use streams.
        # To open another connection, we change the agent's ip.
        daemon._job.agent_info[0].ip = "127.0.0.2"
        daemon._job.agent_info.append(_AgentInfo("127.0.0.1", [1]))

        agent2 = OobleckAgent()
        await agent2.connect_to_master("localhost", daemon.port)
        await agent2.register_agent()

        agent.on_receive_reconfiguration = AsyncMock(
            wraps=agent.on_receive_reconfiguration
        )

        # disconnect agent2
        agent2.conn_[1].close()
        await asyncio.sleep(0.1)
        agent.on_receive_reconfiguration.assert_called()

    @staticmethod
    def fake_worker_main(
        my_ip: str,
        path: Path,
        local_rank: int,
        num_gpus_per_node: int,
        pipe: connection.Connection,
        args: OobleckArguments,
    ):
        factory = OobleckStaticClassFactory(args.model_name, path)
        with patch("socket.gethostbyname", return_value=my_ip), patch(
            "oobleck.execution.engine.PipelineTemplateGenerator.create_pipeline_templates",
            autospec=True,
            return_value=[
                factory.get_dummy_pipeline_template(
                    num_stages=num_gpus_per_node,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=1,
                )
            ],
        ), patch(
            "oobleck.execution.engine.get_profile_results",
            autospec=True,
            return_value=factory.get_dummy_profile(),
        ), patch(
            "oobleck.execution.engine.OobleckDataset",
            autospec=True,
            return_value=factory.get_dataset(),
        ), patch(
            "oobleck.execution.engine.OobleckModel",
            autospec=True,
            return_value=factory.get_model(),
        ):
            worker_main(local_rank, num_gpus_per_node, pipe, args)

    @pytest.mark.asyncio
    async def test_launch_workers(
        self,
        agent: OobleckAgent,
        model_name_fixture: str,
        tmp_path: Path,
        mocker: MockerFixture,
    ):
        await agent.register_agent()

        dataset_path, dataset_name = datasets[model_name_fixture]
        fake_args = OobleckArguments(
            model_name=model_name_fixture,
            model_tag="elastic_test",
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            fault_threshold=1,
            model_args=model_args[model_name_fixture],
            global_microbatch_size=8,
            steps=10,
        )

        my_ip: str = "127.0.0.1"
        mocker.patch("socket.gethostbyname", return_value=my_ip)

        mocker.patch(
            "oobleck.elastic.agent.worker_main",
            functools.partial(self.fake_worker_main, my_ip, tmp_path),
        )
        await agent.launch_workers(4, fake_args)

        # Assume master already sent dist info
        # Agent can get dist info via:
        # await agent.get_dist_info()

        # Send dist info to workers
        dist_info = message_util.DistributionInfo([my_ip], 4)
        for worker in agent._workers:
            worker.pipe.send(dist_info)

        # Because this agent has rank 0, it should forward worker port
        if dist_info.agent_ips.index(my_ip) == 0:
            await agent.forward_worker_port(agent._workers[0].pipe)

        loop = asyncio.get_running_loop()
        for worker in agent._workers:
            await loop.run_in_executor(None, worker.process.join)

    @staticmethod
    def fake_worker_main_wait_for_signal(*args, **kawrgs):
        signal.sigwait([signal.SIGUSR1])

    @pytest.mark.asyncio
    async def test_signal_to_workers(
        self,
        agent: OobleckAgent,
        mocker: MockerFixture,
    ):
        await agent.register_agent()

        mocker.patch(
            "oobleck.elastic.agent.worker_main",
            self.fake_worker_main_wait_for_signal,
        )
        fake_args = OobleckArguments(
            model_name="",
            model_tag="",
            dataset_path="",
            dataset_name=None,
            fault_threshold=1,
            model_args=None,
            global_microbatch_size=0,
            steps=0,
        )
        await agent.launch_workers(4, fake_args)

        mocker.patch("oobleck.elastic.agent.message_util.recv", return_value=[2])
        await agent.on_receive_reconfiguration()

        futures: list[Future] = []
        with ThreadPoolExecutor(len(agent._workers)) as executor:
            for worker in agent._workers:
                futures.append(executor.submit(worker.process.join))

        _, not_done = wait(futures, timeout=30)
        assert len(not_done) == 0
