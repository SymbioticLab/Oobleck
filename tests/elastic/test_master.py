import asyncio
import copy
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import oobleck.elastic.message_util as message_util
from oobleck.elastic.master import OobleckMasterDaemon, _AgentInfo, _Job


class TestOobleckMasterDaemonClass:
    @pytest_asyncio.fixture(autouse=True)
    async def daemon(self, event_loop: asyncio.AbstractEventLoop):
        daemon = await OobleckMasterDaemon.create()
        event_loop.create_task(daemon.run())

        yield daemon

        if not daemon._server.is_serving():
            return
        daemon._server.close()
        await daemon._server.wait_closed()

    @pytest.fixture
    def sample_job(self) -> _Job:
        return _Job("test", [_AgentInfo("127.0.0.1", [0])])

    @pytest_asyncio.fixture(autouse=True)
    async def conns(self, daemon: OobleckMasterDaemon):
        r, w = await asyncio.open_connection("localhost", daemon._port)
        yield r, w
        w.close()
        await w.wait_closed()

    @pytest.mark.asyncio
    async def test_request_job_fail(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        daemon.request_job_handler = AsyncMock(wraps=daemon.request_job_handler)
        daemon.request_job_handler.assert_not_awaited()

        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)

        # Not providing job information within 5 seconds should return failure.
        assert (await message_util.recv_response(r, timeout=None)) == (
            message_util.Response.FAILURE,
            message_util.RequestType.LAUNCH_JOB,
        )

        daemon.request_job_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_job(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns

        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)
        await message_util.send(w, sample_job, need_pickle=True, close=False)

        result = await message_util.recv_response(r)
        assert result == (
            message_util.Response.SUCCESS,
            message_util.RequestType.LAUNCH_JOB,
        )

        w.close()
        await w.wait_closed()

        assert daemon._job
        assert daemon._job.name == sample_job.name

    @pytest.mark.asyncio
    async def test_get_dist_info_fail_no_job(
        self,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)

        with pytest.raises(asyncio.IncompleteReadError):
            await message_util.recv_response(r)

    @pytest.mark.asyncio
    async def test_get_dist_info(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns

        daemon._job = sample_job
        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        await message_util.recv_response(r)

        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)

        assert await message_util.recv_response(r) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.GET_DIST_INFO,
        )

        dist_info: message_util.DistributionInfo = await message_util.recv(
            r, need_pickle=True
        )
        assert len(dist_info.agent_ips) == 1
        assert dist_info.agent_ips[0] == "127.0.0.1"
        assert dist_info.world_size == 1

    @pytest.mark.asyncio
    async def test_get_dist_info_blocked(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns

        # Make ips have two to simulate two nodes must call get_dist_info()
        # to get information.
        sample_job: _Job = copy.deepcopy(sample_job)
        sample_job.agent_info = [
            _AgentInfo("127.0.0.1", [0]),
            _AgentInfo("127.0.0.2", [1]),
        ]

        daemon._job = sample_job
        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        await message_util.recv_response(r)

        # The node calling get_dist_info() should be blocked
        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)

        with pytest.raises(asyncio.TimeoutError):
            await message_util.recv_response(r)

    @pytest.mark.asyncio
    async def test_get_dist_info_by_multiple_clients(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns

        sample_job: _Job = copy.deepcopy(sample_job)
        sample_job.agent_info = [
            _AgentInfo("127.0.0.1", [0]),
            _AgentInfo("127.0.0.2", [1]),
        ]

        daemon._job = sample_job
        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        await message_util.recv_response(r)

        r2, w2 = await asyncio.open_connection("localhost", daemon.port)
        await message_util.send_request_type(
            w2, message_util.RequestType.REGISTER_AGENT
        )
        await message_util.recv_response(r2)

        # First client
        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)
        task = asyncio.create_task(message_util.recv_response(r))

        await asyncio.sleep(2)
        assert len(daemon._nodes_to_rendezvous) == 1
        assert not task.done(), "First client must be blocked."

        # Second client
        await message_util.send_request_type(w2, message_util.RequestType.GET_DIST_INFO)

        # Both must succeed
        while not task.done():
            await asyncio.sleep(0.1)
        assert task.result() == (
            message_util.Response.SUCCESS,
            message_util.RequestType.GET_DIST_INFO,
        )
        assert (await message_util.recv_response(r2)) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.GET_DIST_INFO,
        )

        # Both must receive the same information
        dist_info: message_util.DistributionInfo = await message_util.recv(
            r, need_pickle=True
        )
        dist_info2: message_util.DistributionInfo = await message_util.recv(
            r2, need_pickle=True
        )
        assert dist_info == dist_info2

        w2.close()
        await w2.wait_closed()

    @pytest.mark.asyncio
    async def test_register_agent(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns
        daemon._job = sample_job

        assert daemon._job.agent_info[0].streams is None

        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        assert await message_util.recv_response(r) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.REGISTER_AGENT,
        )
        assert daemon._job.agent_info[0].streams

    @pytest.mark.asyncio
    async def test_ping_fail(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns
        daemon._job = sample_job

        # Check agent is not registered
        assert not daemon._job.agent_info[0].streams

        await message_util.send_request_type(w, message_util.RequestType.PING)

        # Without agent registration, expected to fail
        with pytest.raises(asyncio.IncompleteReadError):
            await message_util.recv_response(r)

    @pytest.mark.asyncio
    async def test_ping(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns
        daemon._job = sample_job

        # Register agent then ping
        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        assert (await message_util.recv_response(r)) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.REGISTER_AGENT,
        )

        assert daemon._job.agent_info[0].streams

        # Reuse the connection
        await message_util.send_request_type(w, message_util.RequestType.PING)
        assert (await message_util.recv_response(r, timeout=10)) == (
            message_util.Response.PONG,
            message_util.RequestType.PING,
        )

    @pytest.mark.asyncio
    async def test_agent_disconnect(
        self,
        daemon: OobleckMasterDaemon,
        conns: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        sample_job: _Job,
    ):
        r, w = conns
        daemon._job = sample_job
        daemon._job.agent_info.append(_AgentInfo("127.0.0.1", [1]))

        daemon.agent_handler = AsyncMock(wraps=daemon.agent_handler)
        daemon.close_agent = AsyncMock(wraps=daemon.close_agent)

        # Register agent streams (technically same with registering agent)
        await message_util.send_request_type(w, message_util.RequestType.REGISTER_AGENT)
        assert await message_util.recv_response(r) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.REGISTER_AGENT,
        )

        # FIXME: currently daemon uses IP for identifier, thus always the first `_AgentInfo``
        # will be chosen during registration.
        # For simulating two agents are registered, move streams to the second _AgentInfo.
        daemon._job.agent_info[1].streams = daemon._job.agent_info[0].streams
        daemon._job.agent_info[0].streams = None

        # create another agent
        r2, w2 = await asyncio.open_connection("localhost", daemon._port)
        await message_util.send_request_type(
            w2, message_util.RequestType.REGISTER_AGENT
        )
        assert await message_util.recv_response(r2) == (
            message_util.Response.SUCCESS,
            message_util.RequestType.REGISTER_AGENT,
        )

        second_agent = daemon._job.agent_info[0]

        # Disconnect an agent
        w2.close()
        await w2.wait_closed()

        assert (await message_util.recv_response(r, timeout=None)) == (
            message_util.Response.RECONFIGURATION,
            message_util.RequestType.UNDEFINED,
        )
        daemon.close_agent.assert_called_once_with(second_agent)
        assert len(daemon._job.agent_info) == 1
