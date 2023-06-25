import asyncio
import unittest.mock

import pytest
import pytest_asyncio

from oobleck.elastic.master import OobleckMasterDaemon


class TestOobleckMasterDaemonClass:
    @pytest_asyncio.fixture(autouse=True)
    async def daemon(self, event_loop: asyncio.AbstractEventLoop):
        daemon = await OobleckMasterDaemon.create()
        event_loop.create_task(daemon.run())
        yield daemon
        daemon._server.close()
        await daemon._server.wait_closed()

    @pytest.mark.asyncio
    async def test_server_started(self, daemon: OobleckMasterDaemon):
        assert daemon._server.is_serving()
        r, w = await asyncio.open_connection("localhost", daemon._port)
        w.close()
        await w.wait_closed()
