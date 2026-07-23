"""Public membership service and concurrently reading node client."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

from oobleck.elastic.membership import (
    MembershipSnapshot,
    StaleSequence,
    membership_snapshot_from_payload,
)
from oobleck.elastic.local_ipc import LocalWorkerRelay
from oobleck.elastic.service_base import MasterControlService
from oobleck.elastic.transport import (
    AsyncioTcpControlTransport,
    ControlConnection,
    ControlTransport,
    MessageEnvelope,
)


class NodeAgentClient:
    def __init__(
        self,
        node_id: str,
        gpu_ids: Sequence[str],
        *,
        transport: ControlTransport | None = None,
        heartbeat_interval_s: float = 1.0,
        addresses: Sequence[str] = (),
        on_membership: Callable[[MessageEnvelope], Awaitable[None]] | None = None,
        on_generation_active: Callable[[MessageEnvelope], Awaitable[None]] | None = None,
        local_worker_socket: str | Path | None = None,
    ) -> None:
        if not node_id or not gpu_ids or heartbeat_interval_s <= 0:
            raise ValueError("node_id, gpu_ids, and a positive heartbeat interval are required")
        self.node_id = node_id
        self.gpu_ids = tuple(gpu_ids)
        discovered = tuple(addresses) or tuple(
            sorted(set(socket.gethostbyname_ex(socket.gethostname())[2]))
        )
        self.addresses = discovered or (socket.gethostbyname(socket.gethostname()),)
        for address in self.addresses:
            try:
                socket.getaddrinfo(address, None)
            except socket.gaierror as exc:
                raise ValueError(f"agent address {address!r} is not resolvable") from exc
        self.transport = transport or AsyncioTcpControlTransport()
        self.heartbeat_interval_s = heartbeat_interval_s
        self.on_membership = on_membership
        self.on_generation_active = on_generation_active
        self.incarnation_id = str(uuid.uuid4())
        self.generation = 0
        self.active_generation = 0
        self.sequence = 0
        self.connection: ControlConnection | None = None
        self.snapshot: MembershipSnapshot | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._last_master_sequence = -1
        self._prepared_generation = -1
        self._ready_sent_generation = -1
        self._ready_metadata: tuple[str, str] | None = None
        self._send_lock = asyncio.Lock()
        self._ready_lock = asyncio.Lock()
        self._stopping = False
        self.local_worker_relay = (
            LocalWorkerRelay(
                local_worker_socket,
                node_id,
                expected_workers=len(self.gpu_ids),
                on_generation_ready=self._local_workers_ready,
            )
            if local_worker_socket is not None
            else None
        )

    async def _send_agent_message(
        self,
        message_type: str,
        generation: int,
        payload: Mapping[str, object],
    ) -> None:
        if self.connection is None:
            raise RuntimeError("agent is not connected")
        async with self._send_lock:
            self.sequence += 1
            await self.connection.send(
                MessageEnvelope(
                    1,
                    message_type,
                    self.node_id,
                    self.incarnation_id,
                    self.sequence,
                    generation,
                    payload,
                )
            )

    async def _local_workers_ready(
        self,
        generation: int,
        snapshot_hash: str,
        plan_checksum: str,
        compatibility_digest: str,
    ) -> None:
        snapshot = self.snapshot
        if (
            snapshot is not None
            and generation == snapshot.generation
            and snapshot_hash == snapshot.snapshot_hash
        ):
            self._ready_metadata = (plan_checksum, compatibility_digest)
            await self._send_ready_if_possible(generation)

    async def _send_ready_if_possible(self, generation: int) -> None:
        async with self._ready_lock:
            snapshot = self.snapshot
            if (
                self.connection is None
                or snapshot is None
                or generation != snapshot.generation
                or self._prepared_generation != generation
                or self._ready_sent_generation >= generation
            ):
                return
            if self.local_worker_relay is not None:
                if not self.local_worker_relay.generation_ready(generation):
                    return
                if self._ready_metadata is None:
                    return
                plan_checksum, compatibility_digest = self._ready_metadata
            else:
                plan_checksum, compatibility_digest = snapshot.snapshot_hash, ""
            self._ready_sent_generation = generation
            try:
                await self._send_agent_message(
                    "generation_ready",
                    generation,
                    {
                        "snapshot_hash": snapshot.snapshot_hash,
                        "plan_checksum": plan_checksum,
                        "compatibility_digest": compatibility_digest,
                    },
                )
            except BaseException:
                self._ready_sent_generation = -1
                raise

    def _validate_master_sequence(self, message: MessageEnvelope) -> None:
        if message.agent_id != "master":
            raise ValueError("expected a control message from the master")
        if message.sequence_number <= self._last_master_sequence:
            raise StaleSequence("duplicate or out-of-order master message")
        self._last_master_sequence = message.sequence_number

    async def _consume_membership(self, message: MessageEnvelope) -> MembershipSnapshot:
        if message.message_type != "membership":
            raise ValueError("expected a membership message from the master")
        self._validate_master_sequence(message)
        snapshot = membership_snapshot_from_payload(message.generation, message.payload)
        if snapshot.generation < self.generation:
            raise ValueError("master sent a stale membership generation")
        self.generation = snapshot.generation
        self.snapshot = snapshot
        self._prepared_generation = -1
        self._ready_sent_generation = -1
        self._ready_metadata = None
        if self.local_worker_relay is not None:
            await self.local_worker_relay.publish(message)
        if self.on_membership is not None:
            await self.on_membership(message)
        self._prepared_generation = snapshot.generation
        await self._send_ready_if_possible(snapshot.generation)
        return snapshot

    async def _consume_active(self, message: MessageEnvelope) -> None:
        if message.message_type != "generation_active":
            raise ValueError("expected a generation_active message")
        self._validate_master_sequence(message)
        if message.generation < self.generation:
            return
        snapshot = self.snapshot
        if snapshot is None:
            raise ValueError("master activated a generation before membership")
        ready_metadata = self._ready_metadata or (snapshot.snapshot_hash, "")
        if message.generation != snapshot.generation or message.payload != {
            "snapshot_hash": snapshot.snapshot_hash,
            "plan_checksum": ready_metadata[0],
            "compatibility_digest": ready_metadata[1],
        }:
            raise ValueError("master activated an unprepared generation")
        self.active_generation = message.generation
        if self.local_worker_relay is not None:
            await self.local_worker_relay.publish(message)
        if self.on_generation_active is not None:
            await self.on_generation_active(message)

    async def connect(self, host: str, port: int) -> MessageEnvelope:
        self._host, self._port = host, port
        if self.local_worker_relay is not None and self.local_worker_relay._server is None:
            await self.local_worker_relay.start()
        self.connection = await self.transport.connect(host, port)
        await self.connection.send(
            MessageEnvelope(
                1,
                "register",
                self.node_id,
                self.incarnation_id,
                self.sequence,
                self.generation,
                {
                    "addresses": list(self.addresses),
                    "gpu_ids": list(self.gpu_ids),
                },
            )
        )
        membership = await self.connection.receive()
        await self._consume_membership(membership)
        return membership

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            await self._send_agent_message("heartbeat", self.generation, {})

    async def _receive_loop(self) -> None:
        assert self.connection is not None
        while True:
            message = await self.connection.receive()
            if message.message_type == "drain_command":
                if message.payload != {"node_id": self.node_id}:
                    raise ValueError("drain command targets a different node")
                await self.drain()
                return
            if message.message_type == "membership":
                await self._consume_membership(message)
            elif message.message_type == "generation_active":
                await self._consume_active(message)
            else:
                raise ValueError(f"unsupported master message {message.message_type!r}")

    async def run(self) -> None:
        if self.connection is None:
            raise RuntimeError("connect() must be called before run()")
        tasks = {
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._receive_loop()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()

    async def run_reconnecting(self, *, retry_delay_s: float = 0.1) -> None:
        """Maintain one incarnation at a time and reconnect after stream loss."""

        if self._host is None or self._port is None:
            raise RuntimeError("connect() must be called before run_reconnecting()")
        if retry_delay_s <= 0:
            raise ValueError("retry_delay_s must be positive")
        while not self._stopping:
            try:
                await self.run()
            except (OSError, ConnectionError, asyncio.IncompleteReadError):
                if self._stopping:
                    break
                if self.connection is not None:
                    with contextlib.suppress(Exception):
                        await self.connection.close()
                self.connection = None
                self.incarnation_id = str(uuid.uuid4())
                self.sequence = 0
                await asyncio.sleep(retry_delay_s)
                await self.connect(self._host, self._port)

    async def drain(self) -> None:
        if self.connection is None:
            raise RuntimeError("agent is not connected")
        self._stopping = True
        await self._send_agent_message("drain", self.generation, {})

    async def close(self, *, graceful: bool = False) -> None:
        self._stopping = True
        if self.connection is not None:
            if graceful:
                await self.drain()
            await self.connection.close()
            self.connection = None
        if self.local_worker_relay is not None:
            await self.local_worker_relay.close()


async def inspect_membership(
    host: str,
    port: int,
    *,
    transport: ControlTransport | None = None,
) -> MembershipSnapshot:
    selected = transport or AsyncioTcpControlTransport()
    connection = await selected.connect(host, port)
    try:
        identity = str(uuid.uuid4())
        await connection.send(MessageEnvelope(1, "inspect", "operator", identity, 0, 0, {}))
        message = await connection.receive()
        client = NodeAgentClient("operator", ("inspection",), transport=selected)
        return await client._consume_membership(message)
    finally:
        await connection.close()


async def request_drain(
    host: str,
    port: int,
    node_id: str,
    *,
    transport: ControlTransport | None = None,
) -> int:
    """Ask the active node incarnation to submit its own graceful drain."""

    selected = transport or AsyncioTcpControlTransport()
    connection = await selected.connect(host, port)
    try:
        identity = str(uuid.uuid4())
        await connection.send(
            MessageEnvelope(
                1,
                "request_drain",
                "operator",
                identity,
                0,
                0,
                {"node_id": node_id},
            )
        )
        reply = await connection.receive()
        if reply.message_type != "drain_accepted" or reply.payload != {"node_id": node_id}:
            raise ValueError("master rejected or malformed the drain request")
        return reply.generation
    finally:
        await connection.close()


__all__ = [
    "MasterControlService",
    "NodeAgentClient",
    "inspect_membership",
    "request_drain",
]
