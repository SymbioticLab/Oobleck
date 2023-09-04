import asyncio
import multiprocessing

import pytest
from pytest_mock import MockerFixture

from oobleck.elastic.agent import OobleckAgent, OobleckArguments, Worker
from oobleck.elastic.master import OobleckMasterDaemon
from tests.elastic.conftest import OobleckElasticTestCase


class TestOobleckAgentClass(OobleckElasticTestCase):
    @pytest.fixture(autouse=True)
    def setup_method(self, mocker: MockerFixture):
        mocker.patch(
            "asyncio.StreamWriter.get_extra_info",
            return_value=(self.sample_ip, "12345"),
        )

    @pytest.mark.asyncio
    async def test_register_agent(
        self,
        daemon: OobleckMasterDaemon,
        agent: OobleckAgent,
    ):
        await agent._register_agent(0)
        await asyncio.sleep(1)
        assert self.sample_ip in [
            connection[0] for connection in daemon._agent_connections
        ]

    @pytest.mark.asyncio
    async def test_launch_workers(
        self,
        daemon: OobleckMasterDaemon,
        agent: OobleckAgent,
        mocker: MockerFixture,
    ):
        mocker.patch("oobleck.elastic.agent.worker_main", new_callable=lambda: 0)

        args_from_master = await agent._register_agent(0)
        await agent._launch_workers(args_from_master)
        assert len(agent._workers) == args_from_master.dist.num_workers
        for worker in agent._workers:
            worker.process.join()

    @staticmethod
    async def agent_process_fn(args: OobleckArguments):
        agent = OobleckAgent(args)
        await agent._connect_to_master(args.dist.master_ip, args.dist.master_port)
        await agent._register_agent()
        await asyncio.sleep(1)
        agent._conn[1].close()

    @pytest.mark.asyncio
    async def test_receive_reconfiguration(
        self, daemon: OobleckMasterDaemon, agent: OobleckAgent, mocker: MockerFixture
    ):
        job_id = 0

        args_from_master = await agent._register_agent(job_id)
        assert args_from_master == daemon._job_arguments[job_id]
        agent._args = args_from_master

        # Fake worker processes
        pipe = multiprocessing.Pipe()
        for i in range(args_from_master.dist.num_workers):
            agent._workers.append(Worker(pipe[1], None))
        pipe_spy = mocker.spy(agent._workers[0].pipe, "send")

        expected_lost_node = "127.0.0.2"

        # Create a new agent, register it, and terminate it
        new_agent = OobleckAgent(
            agent._args.dist.master_ip, agent._args.dist.master_port, job_id
        )
        await new_agent._connect_to_master(
            agent._args.dist.master_ip, agent._args.dist.master_port
        )
        mocker.patch(
            "asyncio.StreamWriter.get_extra_info", return_value=("127.0.0.2", "12345")
        )
        await new_agent._register_agent(0)
        new_agent._conn[1].close()
        await new_agent._conn[1].wait_closed()

        asyncio.create_task(agent.on_receive_response())

        # Yield context so that agent can receive reconfiguration message
        while "127.0.0.2" in agent._args.dist.node_ips:
            await asyncio.sleep(0.1)

        pipe_spy.assert_called_with(expected_lost_node)
