"""Agent-owned local GPU worker process lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from oobleck.config import AgentConfig
from oobleck.elastic.service_public_base import NodeAgentClient
from oobleck.elastic.transport import MessageEnvelope


class LocalWorkerSupervisor:
    """Launch and retire one training process for each configured local GPU."""

    def __init__(
        self,
        script: str | Path,
        script_args: Sequence[str],
        *,
        node_id: str,
        gpu_ids: Sequence[str],
        socket_path: str | Path,
    ) -> None:
        """Capture worker entrypoint, stable node identity, GPUs, and relay path."""

        self.script = Path(script)
        self.script_args = tuple(script_args)
        self.node_id = node_id
        self.gpu_ids = tuple(gpu_ids)
        self.socket_path = Path(socket_path)
        self.processes: list[asyncio.subprocess.Process] = []

    async def start(self) -> None:
        """Spawn one isolated process per GPU with deterministic worker metadata."""

        if self.processes:
            raise RuntimeError("local workers are already running")
        if not self.script.is_file():
            raise FileNotFoundError(f"worker script does not exist: {self.script}")
        for local_rank, gpu_id in enumerate(self.gpu_ids):
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_id,
                    "LOCAL_RANK": "0",
                    "OOBLECK_GPU_ID": gpu_id,
                    "OOBLECK_LOCAL_RANK": str(local_rank),
                    "OOBLECK_TP_SIZE": str(len(self.gpu_ids)),
                    "OOBLECK_LOCAL_WORKER_SOCKET": str(self.socket_path),
                    "OOBLECK_NODE_ID": self.node_id,
                    "OOBLECK_WORKER_ID": f"{self.node_id}:gpu-{gpu_id}",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.script),
                *self.script_args,
                env=environment,
            )
            self.processes.append(process)

    async def wait(self) -> tuple[int, ...]:
        """Wait for every worker and fail the node service if any exits nonzero."""

        if not self.processes:
            raise RuntimeError("local workers have not been started")
        codes = tuple(await asyncio.gather(*(item.wait() for item in self.processes)))
        failed = [code for code in codes if code != 0]
        if failed:
            raise RuntimeError(f"local GPU workers exited unsuccessfully: {codes}")
        return codes

    async def close(self) -> None:
        """Terminate workers, escalate to kill after a bound, and forget handles."""

        running = [item for item in self.processes if item.returncode is None]
        for process in running:
            process.terminate()
        if running:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(item.wait() for item in running)), timeout=5.0
                )
            except TimeoutError:
                for process in running:
                    if process.returncode is None:
                        process.kill()
                await asyncio.gather(*(item.wait() for item in running))
        self.processes.clear()


async def run_agent_service(
    config: AgentConfig,
    *,
    on_membership: Callable[[MessageEnvelope], Awaitable[None]] | None = None,
    on_generation_active: Callable[[MessageEnvelope], Awaitable[None]] | None = None,
) -> None:
    """Own the complete node-agent and optional GPU-worker process lifetime.

    The agent registers before workers start so membership and local IPC are available. Without a
    worker script it simply maintains the reconnecting lease. With workers, the agent stream and
    process cohort run concurrently; unsuccessful workers fail the node, while clean worker completion
    requests a graceful drain. The ``finally`` path always retires processes, closes transports, and
    cancels the remaining agent task.
    """

    client = NodeAgentClient(
        config.node_id,
        config.gpu_ids,
        heartbeat_interval_s=config.heartbeat_interval_s,
        addresses=config.addresses,
        on_membership=on_membership,
        on_generation_active=on_generation_active,
        local_worker_socket=config.local_worker_socket,
    )
    supervisor = (
        LocalWorkerSupervisor(
            config.worker_script,
            config.worker_args,
            node_id=config.node_id,
            gpu_ids=config.gpu_ids,
            socket_path=config.local_worker_socket,
        )
        if config.worker_script is not None and config.local_worker_socket is not None
        else None
    )
    agent_task: asyncio.Task[None] | None = None
    workers_completed = False
    await client.connect(config.master_host, config.master_port)
    try:
        if supervisor is None:
            await client.run_reconnecting()
            return
        await supervisor.start()
        agent_task = asyncio.create_task(client.run_reconnecting())
        await supervisor.wait()
        workers_completed = True
    finally:
        if supervisor is not None:
            await supervisor.close()
        with contextlib.suppress(Exception):
            await client.close(graceful=workers_completed and not client._stopping)
        if agent_task is not None:
            agent_task.cancel()
            await asyncio.gather(agent_task, return_exceptions=True)


__all__ = ["LocalWorkerSupervisor", "run_agent_service"]
