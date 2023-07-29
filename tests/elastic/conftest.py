import asyncio

import pytest_asyncio

from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import OobleckMasterDaemon, _AgentInfo, _Job


class OobleckElasticTestCase:
    @pytest_asyncio.fixture(autouse=True)
    async def daemon(
        self, event_loop: asyncio.AbstractEventLoop
    ) -> OobleckMasterDaemon:
        daemon = await OobleckMasterDaemon.create()
        event_loop.create_task(daemon.run())

        yield daemon

        if not daemon._server.is_serving():
            return
        daemon._server.close()
        await daemon._server.wait_closed()

    @pytest_asyncio.fixture
    async def agent(self, daemon: OobleckMasterDaemon) -> OobleckAgent:
        daemon._job = _Job("test", [_AgentInfo("127.0.0.1", [0])])

        agent = OobleckAgent()
        await agent.connect_to_master("localhost", daemon.port)
        yield agent
        agent.conn_[1].close()
        await agent.conn_[1].wait_closed()
