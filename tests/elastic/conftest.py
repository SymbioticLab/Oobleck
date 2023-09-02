import asyncio

import pytest_asyncio
from pytest_mock import MockerFixture

from oobleck.elastic.agent import OobleckAgent
from oobleck.elastic.master import OobleckMasterDaemon
from oobleck.elastic.training_util import (
    DistributedArguments,
    JobArguments,
    ModelArguments,
    OobleckArguments,
)


class OobleckElasticTestCase:
    sample_num_workers: int = 4
    sample_ip: str = "127.0.0.1"

    @pytest_asyncio.fixture(autouse=True)
    async def daemon(
        self,
        event_loop: asyncio.AbstractEventLoop,
    ) -> OobleckMasterDaemon:
        daemon = await OobleckMasterDaemon.create("127.0.0.1", 0)
        daemon._job_arguments[0] = OobleckArguments(
            dist=DistributedArguments(
                master_ip="127.0.0.1",
                master_port=daemon.port,
                node_ips=["127.0.0.1", "127.0.0.2"],
                username="test",
            ),
            job=JobArguments(),
            model=ModelArguments(
                model_name="gpt2",
                model_tag="test",
                dataset_path="test",
                dataset_name=None,
            ),
        )

        event_loop.create_task(daemon._server.serve_forever())

        yield daemon

        if not daemon._server.is_serving():
            return
        daemon._server.close()
        await daemon._server.wait_closed()

    @pytest_asyncio.fixture
    async def agent(
        self, daemon: OobleckMasterDaemon, mocker: MockerFixture
    ) -> OobleckAgent:
        agent = OobleckAgent("127.0.0.1", daemon.port, 0)

        future = asyncio.Future()
        future.set_result(None)
        mocker.patch.object(agent, "_run_profiler", return_value=future)
        await agent._connect_to_master("127.0.0.1", daemon.port)
        yield agent
        if not agent._conn[1].is_closing():
            agent._conn[1].close()
            await agent._conn[1].wait_closed()
