"""One-node master/agent/worker deployment, executable from a source checkout."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oobleck.config import AgentConfig
from oobleck.elastic import MasterControlService, run_agent_service


async def run() -> None:
    service = MasterControlService(lease_timeout_s=5, lease_check_interval_s=0.05)
    server = await service.start("127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    worker_script = Path(__file__).with_name("pretrain_llm.py")
    with tempfile.TemporaryDirectory(prefix="oobleck-local-") as directory:
        config = AgentConfig(
            node_id="local-node",
            master_host="127.0.0.1",
            master_port=port,
            gpu_ids=("0",),
            local_worker_socket=Path(directory) / "agent.sock",
            worker_script=worker_script,
            heartbeat_interval_s=0.1,
        )
        try:
            await run_agent_service(config)
        finally:
            await service.close()
    print(
        f"local deployment completed; final_generation={service.membership.generation}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(run())
