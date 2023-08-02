import asyncio

import pytest
import pytest_asyncio

from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import OobleckMasterDaemon
from oobleck.elastic.training_util import DistributedJobConfiguration, OobleckArguments


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
        agent = OobleckAgent()
        await agent.connect_to_master("localhost", daemon.port)
        yield agent
        agent.conn_[1].close()
        await agent.conn_[1].wait_closed()

    @pytest.fixture
    def sample_job(self, daemon: OobleckMasterDaemon) -> DistributedJobConfiguration:
        return DistributedJobConfiguration(
            master_ip="127.0.0.1",
            master_port=daemon.port,
            node_ips=["127.0.0.1", "127.0.0.2"],
            job_args=OobleckArguments(
                model_name="gpt2",
                model_tag="test",
                dataset_path="test",
                dataset_name=None,
            ),
            username="test",
        )
