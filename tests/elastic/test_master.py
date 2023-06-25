import asyncio
import pickle
from typing import Tuple
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from oobleck.elastic.master import Job, OobleckMasterDaemon, RequestType, Response


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

        w.write(RequestType.LAUNCH_JOB.value.to_bytes(1, "little"))
        await w.drain()

        # Not providing job information within 10 seconds should return failure.
        result = Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == Response.FAILURE

        daemon.request_job_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_job(
        self,
        daemon: OobleckMasterDaemon,
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        job = Job("test", [], 1)

        w.write(RequestType.LAUNCH_JOB.value.to_bytes(1, "little"))
        w.write(pickle.dumps(job))
        w.write_eof()
        await w.drain()

        result = Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == Response.SUCCESS

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

        w.write(RequestType.GET_DIST_INFO.value.to_bytes(1, "little"))
        await w.drain()

        result = Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == Response.FAILURE

    @pytest.mark.asyncio
    async def test_get_dist_info(
        self,
        daemon: OobleckMasterDaemon,
        conns: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ):
        r, w = conns

        daemon._job = Job("test", [], 1)

        w.write(RequestType.GET_DIST_INFO.value.to_bytes(1, "little"))
        await w.drain()

        result = Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == Response.SUCCESS

        job: Job = pickle.loads(await r.read())

        assert job.name == daemon._job.name
