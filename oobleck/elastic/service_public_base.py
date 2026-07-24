"""Public membership service and concurrently reading node client."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from dataclasses import dataclass
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
from oobleck.types import (
    OobleckExecutionPlan,
    PipelineInstance,
    PipelineTemplate,
    stable_rank_map,
)


def _control_only_plan(snapshot: MembershipSnapshot) -> OobleckExecutionPlan:
    """Build a deterministic placeholder when an agent has no GPU-worker relay."""

    node_ids = tuple(node.agent_id for node in snapshot.nodes)
    tp = len(snapshot.nodes[0].gpu_ids)
    rank_map = stable_rank_map(node_ids, tp)
    template = PipelineTemplate(
        "control-only",
        tuple((index, index + 1) for index in range(len(node_ids))),
        tp,
        0.0,
        0.0,
    )
    instance = PipelineInstance(
        "control-only",
        template,
        node_ids,
        tuple(dict(rank_map)[node_id] for node_id in node_ids),
        0,
    )
    previous = (
        snapshot.previous_execution_plan.generation
        if snapshot.previous_execution_plan is not None
        else (snapshot.generation - 1 if snapshot.generation else None)
    )
    return OobleckExecutionPlan(snapshot.generation, (instance,), rank_map, previous)


@dataclass(frozen=True, slots=True)
class ControlStatus:
    """Operator view of proposal membership and activation barrier progress."""

    snapshot: MembershipSnapshot
    active_generation: int
    prepared_agents: tuple[str, ...]
    ready_agents: tuple[str, ...]

    @property
    def generation(self) -> int:
        """Return the newest proposed membership generation."""

        return self.snapshot.generation

    @property
    def active(self) -> bool:
        """Report whether the proposal has completed both activation barriers."""

        return self.active_generation == self.snapshot.generation


class NodeAgentClient:
    """Reconnectable node stream plus local GPU-worker barrier aggregation."""

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
        """Initialize one reconnectable node incarnation and its generation state machine.

        Advertised GPU inventory and heartbeat cadence are validated, while addresses are either
        supplied explicitly or discovered and checked for resolvability. Independent cursors track
        master sequencing, membership, preparation, rendezvous, readiness, and activation. Optional
        local IPC aggregates exactly one worker per GPU before node-level acknowledgements, and send
        locks preserve the incarnation sequence across heartbeat and phase tasks.
        """

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
        self._prepared_sent_generation = -1
        self._rendezvous_generation = -1
        self._ready_sent_generation = -1
        self._phase_metadata: tuple[str, str, dict[str, object]] | None = None
        self._send_lock = asyncio.Lock()
        self._ready_lock = asyncio.Lock()
        self._stopping = False
        self.local_worker_relay = (
            LocalWorkerRelay(
                local_worker_socket,
                node_id,
                expected_workers=len(self.gpu_ids),
                on_generation_prepared=self._local_workers_prepared,
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
        """Serialize one sequenced agent message on the current incarnation."""

        if self.connection is None:
            raise RuntimeError("agent is not connected")
        async with self._send_lock:
            self.sequence += 1
            await self.connection.send(
                MessageEnvelope(
                    message_type,
                    self.node_id,
                    self.incarnation_id,
                    self.sequence,
                    generation,
                    payload,
                )
            )

    async def _local_workers_prepared(
        self,
        generation: int,
        snapshot_hash: str,
        plan_checksum: str,
        compatibility_digest: str,
        execution_plan: dict[str, object],
    ) -> None:
        """Record unanimous local preparation and attempt the master acknowledgement."""

        snapshot = self.snapshot
        if (
            snapshot is not None
            and generation == snapshot.generation
            and snapshot_hash == snapshot.snapshot_hash
        ):
            plan = OobleckExecutionPlan.from_dict(execution_plan)
            if plan.plan_checksum != plan_checksum:
                raise ValueError("local prepared plan checksum is inconsistent")
            self._phase_metadata = (plan_checksum, compatibility_digest, plan.to_dict())
            await self._send_prepared_if_possible(generation)

    async def _local_workers_ready(
        self,
        generation: int,
        snapshot_hash: str,
        plan_checksum: str,
        compatibility_digest: str,
    ) -> None:
        """Forward unanimous local readiness only when it matches preparation."""

        snapshot = self.snapshot
        if (
            snapshot is not None
            and generation == snapshot.generation
            and snapshot_hash == snapshot.snapshot_hash
        ):
            if self._phase_metadata is None or self._phase_metadata[:2] != (
                plan_checksum,
                compatibility_digest,
            ):
                raise ValueError("local ready metadata differs from local preparation")
            await self._send_ready_if_possible(generation)

    async def _send_prepared_if_possible(self, generation: int) -> None:
        """Send node-level preparation exactly once when all prerequisites agree.

        The current connection, snapshot, local preparation cursor, and generation must match. With
        local workers, unanimous snapshot/plan/compatibility metadata is required; without them, the
        snapshot hash supplies the compatibility fallback. The sent cursor is rolled back if writing
        fails so a reconnected incarnation may safely acknowledge again.
        """

        async with self._ready_lock:
            snapshot = self.snapshot
            if (
                self.connection is None
                or snapshot is None
                or generation != snapshot.generation
                or self._prepared_generation != generation
                or self._prepared_sent_generation >= generation
            ):
                return
            if self.local_worker_relay is not None:
                if not self.local_worker_relay.generation_prepared(generation):
                    return
                if self._phase_metadata is None:
                    return
                plan_checksum, compatibility_digest, execution_plan = self._phase_metadata
            else:
                execution_plan = _control_only_plan(snapshot).to_dict()
                plan_checksum = str(execution_plan["plan_checksum"])
                compatibility_digest = ""
                self._phase_metadata = (plan_checksum, compatibility_digest, execution_plan)
            self._prepared_sent_generation = generation
            try:
                await self._send_agent_message(
                    "generation_prepared",
                    generation,
                    {
                        "snapshot_hash": snapshot.snapshot_hash,
                        "plan_checksum": plan_checksum,
                        "compatibility_digest": compatibility_digest,
                        "execution_plan": execution_plan,
                    },
                )
            except BaseException:
                self._prepared_sent_generation = -1
                raise

    async def _send_ready_if_possible(self, generation: int) -> None:
        """Send node-level readiness exactly once after matching rendezvous and local WORLD.

        Readiness is gated by the current snapshot, the rendezvous cursor, retained preparation
        metadata, and unanimous local-worker ready state. As with preparation, failure resets the sent
        marker so reconnect can retry without treating a dropped frame as consensus.
        """

        async with self._ready_lock:
            snapshot = self.snapshot
            if (
                self.connection is None
                or snapshot is None
                or generation != snapshot.generation
                or self._rendezvous_generation != generation
                or self._ready_sent_generation >= generation
                or self._phase_metadata is None
            ):
                return
            if self.local_worker_relay is not None and not self.local_worker_relay.generation_ready(
                generation
            ):
                return
            plan_checksum, compatibility_digest, _ = self._phase_metadata
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
        """Reject messages not owned by the master or replayed out of order."""

        if message.agent_id != "master":
            raise ValueError("expected a control message from the master")
        if message.sequence_number <= self._last_master_sequence:
            raise StaleSequence("duplicate or out-of-order master message")
        self._last_master_sequence = message.sequence_number

    async def _consume_membership(self, message: MessageEnvelope) -> MembershipSnapshot:
        """Install membership and invalidate every phase derived from its predecessor.

        Master sequence and snapshot checksum are validated before local state changes.
        Preparation/rendezvous/readiness cursors and compatibility metadata reset together,
        then the complete snapshot is relayed to GPU workers and optional callbacks. The
        agent acknowledges preparation only after every configured local worker agrees.
        """

        if message.message_type != "membership":
            raise ValueError("expected a membership message from the master")
        self._validate_master_sequence(message)
        snapshot = membership_snapshot_from_payload(message.generation, message.payload)
        if snapshot.generation < self.generation:
            raise ValueError("master sent a stale membership generation")
        self.generation = snapshot.generation
        self.snapshot = snapshot
        self._prepared_generation = -1
        self._prepared_sent_generation = -1
        self._rendezvous_generation = -1
        self._ready_sent_generation = -1
        self._phase_metadata = None
        if self.local_worker_relay is not None:
            await self.local_worker_relay.publish(message)
        if self.on_membership is not None:
            await self.on_membership(message)
        self._prepared_generation = snapshot.generation
        await self._send_prepared_if_possible(snapshot.generation)
        return snapshot

    async def _consume_rendezvous(self, message: MessageEnvelope) -> None:
        """Accept rendezvous only for the locally prepared generation and metadata.

        Stale rendezvous is harmlessly ignored; a future or mismatched checksum, plan, or
        compatibility digest is a protocol error. Matching rendezvous is relayed to local
        workers, and the agent reports ready only after their second barrier completes.
        """

        if message.message_type != "generation_rendezvous":
            raise ValueError("expected a generation_rendezvous message")
        self._validate_master_sequence(message)
        if message.generation < self.generation:
            return
        snapshot = self.snapshot
        if snapshot is None or self._phase_metadata is None:
            raise ValueError("master published rendezvous before local preparation")
        if message.generation != snapshot.generation or message.payload != {
            "snapshot_hash": snapshot.snapshot_hash,
            "plan_checksum": self._phase_metadata[0],
            "compatibility_digest": self._phase_metadata[1],
        }:
            raise ValueError("master rendezvous does not match local preparation")
        self._rendezvous_generation = message.generation
        if self.local_worker_relay is not None:
            await self.local_worker_relay.publish(message)
        await self._send_ready_if_possible(message.generation)

    async def _consume_active(self, message: MessageEnvelope) -> None:
        """Complete activation only for the generation this node prepared and readied.

        The master sequence, membership hash, plan checksum, and compatibility digest must
        all match local phase state. Once accepted, activation is broadcast to GPU workers
        before the optional application callback observes the new active generation.
        """

        if message.message_type != "generation_active":
            raise ValueError("expected a generation_active message")
        self._validate_master_sequence(message)
        if message.generation < self.generation:
            return
        snapshot = self.snapshot
        if snapshot is None:
            raise ValueError("master activated a generation before membership")
        ready_metadata = self._phase_metadata or (snapshot.snapshot_hash, "", {})
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
        """Start local IPC, register this incarnation, and consume membership."""

        self._host, self._port = host, port
        if self.local_worker_relay is not None and self.local_worker_relay._server is None:
            await self.local_worker_relay.start()
        self.connection = await self.transport.connect(host, port)
        await self.connection.send(
            MessageEnvelope(
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
        """Renew the master lease on an independent send cadence."""

        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            await self._send_agent_message("heartbeat", self.generation, {})

    async def _receive_loop(self) -> None:
        """Continuously dispatch master phases and targeted drain commands."""

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
            elif message.message_type == "generation_rendezvous":
                await self._consume_rendezvous(message)
            elif message.message_type == "generation_active":
                await self._consume_active(message)
            else:
                raise ValueError(f"unsupported master message {message.message_type!r}")

    async def run(self) -> None:
        """Run heartbeat and receive loops until either stream task fails."""

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
        """Reconnect after transport loss without reusing incarnation ordering state.

        Ordinary protocol/application errors still escape. Recoverable stream errors close the old
        connection, allocate a fresh incarnation ID, reset sequence numbering, wait the configured
        retry delay, and register again. A graceful stop suppresses reconnection entirely.
        """

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
        """Stop reconnecting and request graceful incarnation removal."""

        if self.connection is None:
            raise RuntimeError("agent is not connected")
        self._stopping = True
        await self._send_agent_message("drain", self.generation, {})

    async def close(self, *, graceful: bool = False) -> None:
        """Optionally drain, then close master and local-worker transports."""

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
    """Fetch one immutable membership snapshot over a short-lived operator stream."""

    selected = transport or AsyncioTcpControlTransport()
    connection = await selected.connect(host, port)
    try:
        identity = str(uuid.uuid4())
        await connection.send(MessageEnvelope("inspect", "operator", identity, 0, 0, {}))
        message = await connection.receive()
        client = NodeAgentClient("operator", ("inspection",), transport=selected)
        return await client._consume_membership(message)
    finally:
        await connection.close()


async def inspect_status(
    host: str,
    port: int,
    *,
    transport: ControlTransport | None = None,
) -> ControlStatus:
    """Fetch membership together with prepared/ready/active barrier progress.

    The operator stream is deliberately short lived and never enters membership.
    Its status payload is reconstructed through the same snapshot validator used
    by agents, then paired with the active generation and the agent IDs that have
    reached each pending barrier. The connection is closed on success or failure.
    """

    selected = transport or AsyncioTcpControlTransport()
    connection = await selected.connect(host, port)
    try:
        identity = str(uuid.uuid4())
        await connection.send(MessageEnvelope("inspect_status", "operator", identity, 0, 0, {}))
        message = await connection.receive()
        if message.message_type != "status":
            raise ValueError("master returned an invalid status response")
        payload = message.payload
        snapshot = membership_snapshot_from_payload(
            message.generation,
            {
                "nodes": payload["nodes"],
                "removed_nodes": payload["removed_nodes"],
                "added_nodes": payload["added_nodes"],
                "detection_seconds": payload["detection_seconds"],
                "snapshot_hash": payload["snapshot_hash"],
                "previous_execution_plan": None,
            },
        )
        return ControlStatus(
            snapshot,
            int(payload["active_generation"]),
            tuple(str(item) for item in payload["prepared_agents"]),
            tuple(str(item) for item in payload["ready_agents"]),
        )
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
    "ControlStatus",
    "MasterControlService",
    "NodeAgentClient",
    "inspect_membership",
    "inspect_status",
    "request_drain",
]
