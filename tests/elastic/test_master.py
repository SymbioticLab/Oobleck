import asyncio
import pickle
from typing import Tuple
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import oobleck.elastic.message_util as message_util
from oobleck.elastic.master import Job, OobleckMasterDaemon


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
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        daemon.request_job_handler = AsyncMock(wraps=daemon.request_job_handler)
        daemon.request_job_handler.assert_not_awaited()

        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)

        # Not providing job information within 5 seconds should return failure.
        result = message_util.Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == message_util.Response.FAILURE

        daemon.request_job_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_job(
        self,
        daemon: OobleckMasterDaemon,
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        job = Job("test", [], 1)

        await message_util.send_request_type(w, message_util.RequestType.LAUNCH_JOB)
        await message_util.send(w, job, need_pickle=True, close=False)

        result = await message_util.recv_response(r)
        assert result == message_util.Response.SUCCESS

        w.close()
        await w.wait_closed()

        assert daemon._job
        assert daemon._job.name == job.name

    @pytest.mark.asyncio
    async def test_get_dist_info_fail_no_job(
        self,
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)

        result = await message_util.recv_response(r)
        assert result == message_util.Response.FAILURE

    @pytest.mark.asyncio
    async def test_get_dist_info(
        self,
        daemon: OobleckMasterDaemon,
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        daemon._job = Job("test", [], 1)

        await message_util.send_request_type(w, message_util.RequestType.GET_DIST_INFO)
        await message_util.send(w, daemon._job, need_pickle=True, close=False)

        result = await message_util.recv_response(r)
        assert result == message_util.Response.SUCCESS

        job: Job = await message_util.recv(r, need_pickle=True)

        assert job.name == daemon._job.name
