"""Membership master and reconnecting node client over ControlTransport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import uuid
from dataclasses import asdict
from typing import Awaitable, Callable, Sequence

from oobleck.elastic.membership import MembershipSnapshot, MembershipStateMachine, NodeIdentity
from oobleck.elastic.transport import (
    AsyncioTcpControlTransport,
    ControlConnection,
    ControlTransport,
    MessageEnvelope,
    ProtocolError,
    PROTOCOL_VERSION,
)

from oobleck.types import OobleckExecutionPlan


class MasterControlService:
    def __init__(
        self,
        transport: ControlTransport | None = None,
        *,
        lease_timeout_s: float = 5.0,
        lease_check_interval_s: float = 0.5,
        max_nodes: int | None = None,
        gpu_ids_per_node: int | None = None,
    ) -> None:
        self.transport = transport or AsyncioTcpControlTransport()
        self.membership = MembershipStateMachine(
            lease_timeout_s=lease_timeout_s,
            max_nodes=max_nodes,
            gpu_ids_per_node=gpu_ids_per_node,
        )
        self.lease_check_interval_s = lease_check_interval_s
        self._connections: dict[str, tuple[str, ControlConnection]] = {}
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._lease_task: asyncio.Task[None] | None = None
        self._master_sequence = 0
        self._proposed_snapshot: MembershipSnapshot | None = None
        self._prepared_agents: dict[str, tuple[str, str, str]] = {}
        self._rendezvous_metadata: tuple[str, str] | None = None
        self._proposed_execution_plan: OobleckExecutionPlan | None = None
        self.active_execution_plan: OobleckExecutionPlan | None = None
        self._ready_agents: dict[str, tuple[str, str]] = {}
        self.active_generation = 0

    async def start(self, host: str, port: int) -> asyncio.AbstractServer:
        if self._server is not None:
            raise RuntimeError("master service is already running")
        self._server = await self.transport.start_server(host, port, self._handle)
        self._lease_task = asyncio.create_task(self._expire_leases())
        return self._server

    async def close(self) -> None:
        if self._lease_task is not None:
            self._lease_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lease_task
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        connections = [item[1] for item in self._connections.values()]
        self._connections.clear()
        await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )
        self._server = None

    def _begin_generation(self, snapshot: MembershipSnapshot) -> MembershipSnapshot:
        """Install a proposal and bind the last activated plan into its hash."""

        snapshot = MembershipSnapshot(
            snapshot.generation,
            snapshot.nodes,
            snapshot.reasons,
            previous_execution_plan=self.active_execution_plan,
        )
        self._proposed_snapshot = snapshot
        self._prepared_agents.clear()
        self._ready_agents.clear()
        self._rendezvous_metadata = None
        self._proposed_execution_plan = None
        if not snapshot.nodes:
            self.active_generation = snapshot.generation
            self.active_execution_plan = None
        return snapshot

    async def _expire_leases(self) -> None:
        while True:
            await asyncio.sleep(self.lease_check_interval_s)
            async with self._lock:
                expired = self.membership.expire_leases()
                for agent_id in expired:
                    self._connections.pop(agent_id, None)
                snapshot = self.membership.publish()
                if snapshot is not None:
                    snapshot = self._begin_generation(snapshot)
            if snapshot is not None:
                await self._broadcast(snapshot)

    async def _handle(self, connection: ControlConnection) -> None:
        identity: NodeIdentity | None = None
        try:
            first = await connection.receive()
            if first.message_type in {"inspect", "inspect_status"}:
                if first.payload:
                    raise ProtocolError(f"{first.message_type} payload must be empty")
                snapshot = self.membership.snapshot()
                reply = (
                    self._membership_message(snapshot)
                    if first.message_type == "inspect"
                    else self._status_message(snapshot)
                )
                await connection.send(reply)
                return
            if first.message_type == "request_drain":
                if set(first.payload) != {"node_id"}:
                    raise ProtocolError("request_drain payload requires node_id")
                node_id = str(first.payload["node_id"])
                async with self._lock:
                    active = self._connections.get(node_id)
                    if active is None:
                        raise ProtocolError(f"node {node_id!r} is not active")
                    self._master_sequence += 1
                    command = MessageEnvelope(
                        PROTOCOL_VERSION,
                        "drain_command",
                        "master",
                        "master",
                        self._master_sequence,
                        self.membership.generation,
                        {"node_id": node_id},
                    )
                    await active[1].send(command)
                await connection.send(
                    MessageEnvelope(
                        PROTOCOL_VERSION,
                        "drain_accepted",
                        "master",
                        "master",
                        self._master_sequence,
                        self.membership.generation,
                        {"node_id": node_id},
                    )
                )
                return
            if first.message_type != "register":
                raise ProtocolError("the first message on an agent stream must be register")
            payload = first.payload
            if set(payload) != {"addresses", "gpu_ids"}:
                raise ProtocolError("register payload requires addresses and gpu_ids")
            identity = NodeIdentity(
                first.agent_id,
                first.incarnation_id,
                tuple(payload["addresses"]),
                tuple(payload["gpu_ids"]),
            )
            async with self._lock:
                self.membership.register(identity, first.sequence_number)
                self._connections[identity.agent_id] = (
                    identity.incarnation_id,
                    connection,
                )
                snapshot = self.membership.publish()
                if snapshot is not None:
                    snapshot = self._begin_generation(snapshot)
            if snapshot is not None:
                await self._broadcast(snapshot)

            while True:
                message = await connection.receive()
                snapshot = None
                rendezvous_message = None
                active_message = None
                async with self._lock:
                    if message.message_type == "heartbeat":
                        self.membership.heartbeat(
                            message.agent_id,
                            message.incarnation_id,
                            message.sequence_number,
                            message.generation,
                        )
                    elif message.message_type == "generation_prepared":
                        proposed = self._proposed_snapshot
                        if (
                            proposed is None
                            or message.generation != proposed.generation
                            or message.payload["snapshot_hash"] != proposed.snapshot_hash
                        ):
                            continue
                        try:
                            execution_plan = OobleckExecutionPlan.from_dict(
                                message.payload["execution_plan"]
                            )
                        except (TypeError, ValueError) as exc:
                            raise ProtocolError("agent submitted an invalid execution plan") from exc
                        if (
                            execution_plan.generation != proposed.generation
                            or execution_plan.plan_checksum
                            != message.payload["plan_checksum"]
                            or (execution_plan.compatibility_digest or "")
                            != message.payload["compatibility_digest"]
                        ):
                            raise ProtocolError("prepared execution plan metadata is inconsistent")
                        metadata = (
                            execution_plan.plan_checksum,
                            execution_plan.compatibility_digest or "",
                            json.dumps(
                                execution_plan.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                        self.membership.acknowledge(
                            message.agent_id,
                            message.incarnation_id,
                            message.sequence_number,
                            message.generation,
                        )
                        if self._prepared_agents and metadata not in set(
                            self._prepared_agents.values()
                        ):
                            raise ProtocolError(
                                "agents disagree on prepared execution plan or runtime compatibility"
                            )
                        self._prepared_agents[message.agent_id] = metadata
                        if set(self._prepared_agents) == set(self.membership.agent_ids):
                            self._proposed_execution_plan = execution_plan
                            self._rendezvous_metadata = metadata[:2]
                            rendezvous_message = self._generation_rendezvous_message(
                                proposed, *metadata[:2]
                            )
                    elif message.message_type == "generation_ready":
                        proposed = self._proposed_snapshot
                        if (
                            proposed is None
                            or message.generation != proposed.generation
                            or message.payload["snapshot_hash"] != proposed.snapshot_hash
                        ):
                            continue
                        metadata = (
                            str(message.payload["plan_checksum"]),
                            str(message.payload["compatibility_digest"]),
                        )
                        if self._rendezvous_metadata is None:
                            raise ProtocolError("agent became ready before rendezvous publication")
                        if metadata != self._rendezvous_metadata:
                            raise ProtocolError(
                                "ready agent disagrees with the prepared generation"
                            )
                        if self._ready_agents and metadata not in set(self._ready_agents.values()):
                            raise ProtocolError(
                                "agents disagree on execution plan or runtime compatibility"
                            )
                        self.membership.acknowledge(
                            message.agent_id,
                            message.incarnation_id,
                            message.sequence_number,
                            message.generation,
                        )
                        self._ready_agents[message.agent_id] = metadata
                        if set(self._ready_agents) == set(self.membership.agent_ids):
                            if self._proposed_execution_plan is None:
                                raise ProtocolError("active generation has no consensus plan")
                            self.active_generation = proposed.generation
                            self.active_execution_plan = self._proposed_execution_plan
                            active_message = self._generation_active_message(proposed, *metadata)
                    elif message.message_type == "drain":
                        self.membership.drain(
                            message.agent_id,
                            message.incarnation_id,
                            message.sequence_number,
                            message.generation,
                        )
                        snapshot = self.membership.publish()
                        if snapshot is not None:
                            snapshot = self._begin_generation(snapshot)
                    else:
                        raise ProtocolError(f"unsupported agent message {message.message_type!r}")
                if rendezvous_message is not None:
                    await self._broadcast_message(rendezvous_message)
                if active_message is not None:
                    await self._broadcast_message(active_message)
                if message.message_type == "drain":
                    if snapshot is not None:
                        await self._broadcast(snapshot)
                    return
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            BrokenPipeError,
            ProtocolError,
        ):
            pass
        finally:
            if identity is not None:
                async with self._lock:
                    active = self._connections.get(identity.agent_id)
                    if active and active[0] == identity.incarnation_id:
                        self._connections.pop(identity.agent_id, None)
                        self.membership.disconnect(identity.agent_id, identity.incarnation_id)
                        snapshot = self.membership.publish()
                        if snapshot is not None:
                            snapshot = self._begin_generation(snapshot)
                    else:
                        snapshot = None
                if snapshot is not None:
                    await self._broadcast(snapshot)

    async def _broadcast(self, snapshot: MembershipSnapshot) -> None:
        message = self._membership_message(snapshot)
        await self._broadcast_message(message)

    async def _broadcast_message(self, message: MessageEnvelope) -> None:
        connections = [item[1] for item in self._connections.values()]
        results = await asyncio.gather(
            *(connection.send(message) for connection in connections),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result

    def _membership_message(self, snapshot: MembershipSnapshot) -> MessageEnvelope:
        self._master_sequence += 1
        return MessageEnvelope(
            PROTOCOL_VERSION,
            "membership",
            "master",
            "master",
            self._master_sequence,
            snapshot.generation,
            {
                "nodes": [asdict(node) for node in snapshot.nodes],
                "reasons": list(snapshot.reasons),
                "snapshot_hash": snapshot.snapshot_hash,
                "previous_execution_plan": (
                    snapshot.previous_execution_plan.to_dict()
                    if snapshot.previous_execution_plan is not None
                    else None
                ),
            },
        )

    def _status_message(self, snapshot: MembershipSnapshot) -> MessageEnvelope:
        self._master_sequence += 1
        return MessageEnvelope(
            PROTOCOL_VERSION,
            "status",
            "master",
            "master",
            self._master_sequence,
            snapshot.generation,
            {
                "nodes": [asdict(node) for node in snapshot.nodes],
                "reasons": list(snapshot.reasons),
                "snapshot_hash": snapshot.snapshot_hash,
                "active_generation": self.active_generation,
                "prepared_agents": sorted(self._prepared_agents),
                "ready_agents": sorted(self._ready_agents),
            },
        )

    def _generation_rendezvous_message(
        self,
        snapshot: MembershipSnapshot,
        plan_checksum: str,
        compatibility_digest: str,
    ) -> MessageEnvelope:
        self._master_sequence += 1
        return MessageEnvelope(
            PROTOCOL_VERSION,
            "generation_rendezvous",
            "master",
            "master",
            self._master_sequence,
            snapshot.generation,
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "plan_checksum": plan_checksum,
                "compatibility_digest": compatibility_digest,
            },
        )

    def _generation_active_message(
        self,
        snapshot: MembershipSnapshot,
        plan_checksum: str,
        compatibility_digest: str,
    ) -> MessageEnvelope:
        self._master_sequence += 1
        return MessageEnvelope(
            PROTOCOL_VERSION,
            "generation_active",
            "master",
            "master",
            self._master_sequence,
            snapshot.generation,
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "plan_checksum": plan_checksum,
                "compatibility_digest": compatibility_digest,
            },
        )


class NodeAgentClient:
    def __init__(
        self,
        node_id: str,
        gpu_ids: Sequence[str],
        *,
        transport: ControlTransport | None = None,
        heartbeat_interval_s: float = 1.0,
        on_membership: Callable[[MessageEnvelope], Awaitable[None]] | None = None,
    ) -> None:
        if not node_id or not gpu_ids:
            raise ValueError("node_id and gpu_ids are required")
        self.node_id = node_id
        self.gpu_ids = tuple(gpu_ids)
        self.transport = transport or AsyncioTcpControlTransport()
        self.heartbeat_interval_s = heartbeat_interval_s
        self.on_membership = on_membership
        self.incarnation_id = str(uuid.uuid4())
        self.generation = 0
        self.sequence = 0
        self.connection: ControlConnection | None = None

    async def connect(self, host: str, port: int) -> MessageEnvelope:
        self.connection = await self.transport.connect(host, port)
        await self.connection.send(
            MessageEnvelope(
                PROTOCOL_VERSION,
                "register",
                self.node_id,
                self.incarnation_id,
                self.sequence,
                self.generation,
                {
                    "addresses": [socket.gethostbyname(socket.gethostname())],
                    "gpu_ids": list(self.gpu_ids),
                },
            )
        )
        membership = await self.connection.receive()
        self.generation = membership.generation
        return membership

    async def run(self) -> None:
        if self.connection is None:
            raise RuntimeError("connect() must be called before run()")
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            self.sequence += 1
            await self.connection.send(
                MessageEnvelope(
                    PROTOCOL_VERSION,
                    "heartbeat",
                    self.node_id,
                    self.incarnation_id,
                    self.sequence,
                    self.generation,
                    {},
                )
            )
            with contextlib.suppress(asyncio.TimeoutError):
                message = await asyncio.wait_for(self.connection.receive(), timeout=0.01)
                if message.message_type == "membership" and message.generation > self.generation:
                    self.generation = message.generation
                    if self.on_membership is not None:
                        await self.on_membership(message)

    async def drain(self) -> None:
        if self.connection is None:
            raise RuntimeError("agent is not connected")
        self.sequence += 1
        await self.connection.send(
            MessageEnvelope(
                PROTOCOL_VERSION, "drain", self.node_id, self.incarnation_id, self.sequence, self.generation, {}
            )
        )
