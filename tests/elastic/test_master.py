import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from oobleck.elastic.master import OobleckMasterDaemon, RequestType, Response


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

    @pytest.mark.asyncio
    async def test_server_started(self, daemon: OobleckMasterDaemon):
        assert daemon._server.is_serving()
        r, w = await asyncio.open_connection("localhost", daemon._port)
        w.close()
        await w.wait_closed()

    @pytest.mark.asyncio
    async def test_request_job(self, daemon: OobleckMasterDaemon):
        daemon.request_job_handler = AsyncMock(wraps=daemon.request_job_handler)

        r, w = await asyncio.open_connection("localhost", daemon._port)

        daemon.request_job_handler.assert_not_awaited()

        w.write(RequestType.LAUNCH_JOB.value.to_bytes(1, "little"))
        await w.drain()

        result = Response(int.from_bytes(await r.readexactly(1), "little"))
        assert result == Response.SUCCESS

        w.close()
        await w.wait_closed()

        daemon.request_job_handler.assert_called_once()
