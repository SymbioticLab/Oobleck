import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture

from oobleck.elastic.agent import OobleckAgent, OobleckAgentArguments, Worker
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
        await agent._register_agent()
        await asyncio.sleep(1)
        assert self.sample_ip in daemon._agent_connections

    @pytest.mark.asyncio
    async def test_fail_register_agent(
        self,
        agent: OobleckAgent,
        mocker: MockerFixture,
    ):
        mocker.patch("asyncio.StreamWriter.get_extra_info", return_value=("0.0.0.0", 0))

        with pytest.raises(ConnectionError):
            await agent._register_agent()

    @pytest.mark.asyncio
    async def test_launch_workers(
        self,
        daemon: OobleckMasterDaemon,
        agent: OobleckAgent,
        mocker: MockerFixture,
    ):
        mocker.patch("oobleck.elastic.agent.worker_main", new_callable=lambda: 0)

        await agent._register_agent()
        await agent._launch_workers(self.sample_num_workers, daemon._job.job_args)
        assert len(agent._workers) == self.sample_num_workers
        for worker in agent._workers:
            worker.process.join()

    @staticmethod
    def agent_process_fn(args: OobleckAgentArguments):
        logging.basicConfig(level=logging.INFO)
        agent = OobleckAgent(args)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            agent._connect_to_master(args.master_ip, args.master_port)
        )
        loop.run_until_complete(agent._register_agent())

    @dataclass
    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

        pid: int

    @pytest.mark.asyncio
    async def test_receive_reconfiguration(
        self, daemon: OobleckMasterDaemon, agent: OobleckAgent, mocker: MockerFixture
    ):
        num_workers = 4

        await agent._register_agent()
        assert list(agent._rank_map.keys()) == daemon._job.node_ips
        # Fake worker processes
        pipe = multiprocessing.Pipe()
        for i in range(num_workers):
            agent._workers.append(Worker(pipe[1], TestOobleckAgentClass.FakeProcess(i)))
        pipe_spy = mocker.spy(agent._workers[0].pipe, "send")
        kill_mock = mocker.patch("os.kill", return_value=None)

        expected_lost_ranks = agent._rank_map["127.0.0.2"]

        with patch(
            "asyncio.StreamWriter.get_extra_info", return_value=("127.0.0.2", "12345")
        ), ProcessPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                executor, TestOobleckAgentClass.agent_process_fn, agent._args
            )
            await asyncio.wait([future])

        # Yield context so that agent can receive reconfiguration message
        while "127.0.0.2" in agent._rank_map:
            await asyncio.sleep(1)

        assert kill_mock.call_count == self.sample_num_workers
        pipe_spy.assert_called_with(expected_lost_ranks)
