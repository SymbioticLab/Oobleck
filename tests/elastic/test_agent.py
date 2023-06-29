import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import oobleck.elastic.message_util as message_util
from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import Job, OobleckMasterDaemon, _AgentInfo


class TestOobleckAgentClassWithNoDaemon:
    pass


class TestOobleckAgentClass:
    @pytest_asyncio.fixture(autouse=True)
    async def daemon(self, event_loop: asyncio.AbstractEventLoop):
        daemon = await OobleckMasterDaemon.create()
        daemon._job = Job("test", [_AgentInfo("127.0.0.1", [0])])
        event_loop.create_task(daemon.run())

        yield daemon

        if not daemon._server.is_serving():
            return
        daemon._server.close()
        await daemon._server.wait_closed()

    @pytest_asyncio.fixture
    async def agent(self, daemon: OobleckMasterDaemon):
        agent = OobleckAgent()
        await agent.connect_to_master("localhost", daemon.port)
        yield agent
        agent.conn_[1].close()
        await agent.conn_[1].wait_closed()

    @pytest.mark.asyncio
    async def test_register_agent(self, agent: OobleckAgent):
        agent.send_request = AsyncMock(wraps=agent.send_request)
        await agent.register_agent()

        await asyncio.sleep(1)
        agent.send_request.assert_called_with(message_util.RequestType.PING, None, None)

    @pytest.mark.asyncio
    async def test_get_dist_info(self, agent: OobleckAgent):
        await agent.register_agent()

        agent.send_request = AsyncMock(wraps=agent.send_request)
        agent.on_receive_dist_info = AsyncMock(wraps=agent.on_receive_dist_info)
        await agent.get_dist_info()

        await asyncio.sleep(0.2)
        agent.send_request.assert_called_with(
            message_util.RequestType.GET_DIST_INFO, None, agent.on_receive_dist_info
        )
        agent.on_receive_dist_info.assert_called()
