import asyncio
import logging
import multiprocessing
import os
import signal
from multiprocessing.connection import Connection
from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture

from oobleck.elastic.agent import OobleckAgent, OobleckAgentArguments
from oobleck.elastic.master import OobleckMasterDaemon
from oobleck.elastic.training_util import OobleckArguments
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
        await asyncio.wait([worker.process for worker in agent._workers])

    @staticmethod
    def fake_worker_main_echo_lost_ranks(
        index: int,
        num_workers: int,
        pipe: Connection,
        args: OobleckArguments,
    ):
        # To notify main process that it is ready
        pipe.send(0)
        signal.sigwait([signal.SIGUSR1])

    @staticmethod
    def agent_process_fn(args: OobleckAgentArguments):
        logging.basicConfig(level=logging.INFO)
        agent = OobleckAgent(args)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            agent._connect_to_master(args.master_ip, args.master_port)
        )
        loop.run_until_complete(agent._register_agent())

    @pytest.mark.asyncio
    async def test_receive_reconfiguration(
        self, daemon: OobleckMasterDaemon, agent: OobleckAgent, mocker: MockerFixture
    ):
        mocker.patch(
            "oobleck.elastic.agent.worker_main",
            new=TestOobleckAgentClass.fake_worker_main_echo_lost_ranks,
        )
        kill_spy = mocker.spy(os, "kill")

        await agent._register_agent()
        await agent._launch_workers(self.sample_num_workers, daemon._job.job_args)
        pipe_spy = mocker.spy(agent._workers[0].pipe, "send")

        assert list(agent._rank_map.keys()) == daemon._job.node_ips
        expected_lost_ranks = agent._rank_map["127.0.0.2"]

        for worker in agent._workers:
            worker.pipe.recv()

        with patch(
            "asyncio.StreamWriter.get_extra_info", return_value=("127.0.0.2", "12345")
        ):
            agent2 = multiprocessing.get_context("spawn").Process(
                target=self.agent_process_fn, args=(agent._args,)
            )
            agent2.start()
            future = asyncio.get_running_loop().run_in_executor(None, agent2.join)
            await asyncio.wait([future])
        await asyncio.sleep(1)

        assert "127.0.0.2" not in agent._rank_map
        assert kill_spy.call_count == self.sample_num_workers
        pipe_spy.assert_called_with(expected_lost_ranks)
