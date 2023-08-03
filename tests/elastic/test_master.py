import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

import oobleck.elastic.message_util as message_util
from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import OobleckMasterDaemon
from oobleck.elastic.training_util import DistributedJobConfiguration
from tests.elastic.conftest import OobleckElasticTestCase

"""
MasterDaemon test cases

1. request job handler (from run.py)
2. launch agent (to OobleckAgent)
3. agent register handler (from OobleckAgent)
4. agent disconnection handler (from OobleckAgent)
"""


class TestOobleckMasterDaemonClass(OobleckElasticTestCase):
    @pytest_asyncio.fixture(autouse=True)
    async def client_conns(
        self, daemon: OobleckMasterDaemon
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        r, w = await asyncio.open_connection("localhost", daemon._port)
        yield r, w
        w.close()
        await w.wait_closed()

    @pytest.mark.asyncio
    async def test_request_job(
        self,
        daemon: OobleckMasterDaemon,
        client_conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        mocker: MockerFixture,
    ):
        job = daemon._job
        # Daemon must not have a job config
        daemon._job = None

        mock_run_agents = mocker.patch.object(
            daemon, "run_node_agent", return_value=AsyncMock(return_value=None)
        )

        r, w = client_conns
        """Cehck if master launches agents."""
        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)
        await message_util.send(w, job, need_pickle=True, close=False)

        result = await message_util.recv_response(r)
        assert result == (
            message_util.Response.SUCCESS,
            message_util.RequestType.LAUNCH_JOB,
        )

        assert mock_run_agents.call_count == len(job.node_ips)
        assert daemon._job == job

        w.close()
        await w.wait_closed()

    @pytest.mark.asyncio
    async def test_request_job_fail(
        self,
        client_conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = client_conns
        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)
        assert (await message_util.recv_response(r)) == (
            message_util.Response.FAILURE,
            message_util.RequestType.LAUNCH_JOB,
        )

    @pytest.mark.asyncio
    async def test_agent_disconnect(
        self,
        daemon: OobleckMasterDaemon,
        mocker: MockerFixture,
    ):
        close_agent_handler_spy = mocker.spy(daemon, "close_agent")

        agents: list[OobleckAgent] = []

        for agent_ip in ["127.0.0.1", "127.0.0.2"]:
            agent = OobleckAgent()
            await agent.connect_to_master("localhost", daemon.port)
            mocker.patch(
                "asyncio.StreamWriter.get_extra_info", return_value=(agent_ip, 0)
            )
            await message_util.send_request_type(
                agent._conn[1], message_util.RequestType.REGISTER_AGENT
            )
            result, _ = await message_util.recv_response(agent._conn[0])
            assert result == message_util.Response.SUCCESS
            agents.append(agent)

        # close the second agent and check if notification goes to the first agent
        agents[1]._conn[1].close()
        await agents[1]._conn[1].wait_closed()

        await asyncio.sleep(1)
        close_agent_handler_spy.assert_called_once_with("127.0.0.2")

        # check first agent receives reconfiguration broadcast
        result, req = await message_util.recv_response(agents[0]._conn[0])
        assert req == message_util.RequestType.UNDEFINED
        assert result == message_util.Response.RECONFIGURATION

        # cleanup
        agents[0]._conn[1].close()
        await agents[0]._conn[1].wait_closed()

    # @pytest.mark.asyncio
    # async def test_ping_fail(
    #     self,
    #     daemon: OobleckMasterDaemon,
    #     conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    #     sample_job: DistributedJobConfiguration,
    # ):
    #     r, w = conns
    #     daemon._job = sample_job

    #     # Check agent is not registered
    #     assert not daemon._job.agent_info[0].streams

    #     await message_util.send_request_type(w, message_util.RequestType.PING)

    #     # Without agent registration, expected to fail
    #     with pytest.raises(asyncio.IncompleteReadError):
    #         await message_util.recv_response(r)

    # @pytest.mark.asyncio
    # async def test_ping(
    #     self,
    #     daemon: OobleckMasterDaemon,
    #     conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    #     sample_job: DistributedJobConfiguration,
    # ):
    #     r, w = conns
    #     daemon._job = sample_job

    #     # Register agent then ping
    #     await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
    #     assert (await message_util.recv_response(r)) == (
    #         message_util.Response.SUCCESS,
    #         message_util.RequestType.REGISTER_AGENT,
    #     )

    #     assert daemon._job.agent_info[0].streams

    #     # Reuse the connection
    #     await message_util.send_request_type(w, message_util.RequestType.PING)
    #     assert (await message_util.recv_response(r, timeout=10)) == (
    #         message_util.Response.PONG,
    #         message_util.RequestType.PING,
    #     )
