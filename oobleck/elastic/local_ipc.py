"""Unix-domain generation barrier between one CPU agent and local GPU workers."""

from __future__ import annotations

import asyncio
import contextlib
import stat
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from oobleck.elastic.membership import (
    MembershipSnapshot,
    membership_snapshot_from_payload,
)
from oobleck.elastic.transport import (
    DEFAULT_MAX_FRAME_BYTES,
    ControlConnection,
    MessageEnvelope,
    ProtocolError,
)

WorkerPhaseCallback = Callable[[int, str, str, str], Awaitable[None]]


class LocalWorkerRelay:
    """Broadcast generation phases and aggregate local GPU-worker consensus."""

    def __init__(
        self,
        path: str | Path,
        node_id: str,
        *,
        expected_workers: int = 1,
        on_generation_prepared: WorkerPhaseCallback | None = None,
        on_generation_ready: WorkerPhaseCallback | None = None,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        """Initialize the Unix listener and per-generation local consensus state."""

        if expected_workers < 1:
            raise ValueError("expected_workers must be positive")
        self.path = Path(path)
        self.node_id = node_id
        self.expected_workers = expected_workers
        self.on_generation_prepared = on_generation_prepared
        self.on_generation_ready = on_generation_ready
        self.max_frame_bytes = max_frame_bytes
        self._server: asyncio.AbstractServer | None = None
        self._workers: dict[str, ControlConnection] = {}
        self._latest_membership: MessageEnvelope | None = None
        self._latest_rendezvous: MessageEnvelope | None = None
        self._latest_active: MessageEnvelope | None = None
        self._prepared: dict[str, tuple[str, str, str]] = {}
        self._acknowledged: dict[str, tuple[str, str, str]] = {}
        self._prepared_notified_generation = -1
        self._ready_notified_generation = -1
        self._lock = asyncio.Lock()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        """Return registered stable worker identities in deterministic order."""

        return tuple(sorted(self._workers))

    def _phase_complete(self, generation: int, values: dict[str, tuple[str, str, str]]) -> bool:
        """Require every configured worker to report against current membership."""

        latest = self._latest_membership
        return (
            latest is not None
            and latest.generation == generation
            and len(self._workers) == self.expected_workers
            and len(values) == self.expected_workers
        )

    def generation_prepared(self, generation: int) -> bool:
        """Report whether all local workers compiled the proposed generation."""

        return self._phase_complete(generation, self._prepared)

    def generation_ready(self, generation: int) -> bool:
        """Report whether all local workers initialized replacement WORLD."""

        return self._phase_complete(generation, self._acknowledged)

    def _phase_metadata(
        self,
        generation: int,
        values: dict[str, tuple[str, str, str]],
        phase: str,
    ) -> tuple[str, str, str]:
        """Return unanimous snapshot, plan, and compatibility identities."""

        if not self._phase_complete(generation, values):
            raise RuntimeError(f"local generation is not {phase}")
        agreed = set(values.values())
        if len(agreed) != 1:
            raise ProtocolError(f"local workers disagree on {phase} generation plan compatibility")
        return next(iter(agreed))

    def preparation_metadata(self, generation: int) -> tuple[str, str, str]:
        """Return agreed metadata for completed local preparation."""

        return self._phase_metadata(generation, self._prepared, "prepared")

    def readiness_metadata(self, generation: int) -> tuple[str, str, str]:
        """Return agreed metadata for completed local readiness."""

        return self._phase_metadata(generation, self._acknowledged, "ready")

    async def start(self) -> None:
        """Safely replace a stale socket and begin accepting local workers."""

        if self._server is not None:
            raise RuntimeError("local worker relay is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            mode = self.path.stat().st_mode
            if not stat.S_ISSOCK(mode):
                raise ValueError(f"refusing to replace non-socket path {self.path}")
            self.path.unlink()

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Wrap one Unix stream in the same framed control protocol."""

            connection = ControlConnection(reader, writer, max_frame_bytes=self.max_frame_bytes)
            await self._handle(connection)

        self._server = await asyncio.start_unix_server(accept, path=self.path)

    async def _handle(self, connection: ControlConnection) -> None:
        """Register one worker incarnation and aggregate its phase acknowledgements."""

        worker_id: str | None = None
        try:
            registration = await connection.receive()
            if registration.message_type != "worker_register":
                raise ProtocolError("local worker must register first")
            if registration.payload != {"node_id": self.node_id}:
                raise ProtocolError("local worker registered for a different node")
            worker_id = registration.agent_id
            async with self._lock:
                previous = self._workers.get(worker_id)
                if previous is None and len(self._workers) >= self.expected_workers:
                    raise ProtocolError("more local workers registered than configured GPUs")
                self._workers[worker_id] = connection
                phases = (
                    self._latest_membership,
                    self._latest_rendezvous,
                    self._latest_active,
                )
            if previous is not None:
                await previous.close()
            for phase in phases:
                if phase is not None:
                    await connection.send(phase)
            while True:
                message = await connection.receive()
                if message.message_type != "worker_ack":
                    raise ProtocolError("local workers may only acknowledge generations")
                if (
                    message.agent_id != worker_id
                    or message.incarnation_id != registration.incarnation_id
                ):
                    raise ProtocolError("invalid local worker phase acknowledgement")
                metadata = (
                    str(message.payload["snapshot_hash"]),
                    str(message.payload["plan_checksum"]),
                    str(message.payload["compatibility_digest"]),
                )
                phase_name = str(message.payload["phase"])
                callback: WorkerPhaseCallback | None = None
                async with self._lock:
                    latest = self._latest_membership
                    if latest is None or message.generation != latest.generation:
                        continue
                    if metadata[0] != latest.payload["snapshot_hash"]:
                        raise ProtocolError("worker acknowledged a different membership snapshot")
                    if phase_name == "prepared":
                        values = self._prepared
                        values[worker_id] = metadata
                        if (
                            self.generation_prepared(message.generation)
                            and self._prepared_notified_generation != message.generation
                        ):
                            agreed = self.preparation_metadata(message.generation)
                            self._prepared_notified_generation = message.generation
                            callback = self.on_generation_prepared
                    else:
                        rendezvous = self._latest_rendezvous
                        if (
                            rendezvous is None
                            or rendezvous.generation != message.generation
                            or rendezvous.payload
                            != {
                                "snapshot_hash": metadata[0],
                                "plan_checksum": metadata[1],
                                "compatibility_digest": metadata[2],
                            }
                        ):
                            raise ProtocolError("worker became ready before matching rendezvous")
                        values = self._acknowledged
                        values[worker_id] = metadata
                        if (
                            self.generation_ready(message.generation)
                            and self._ready_notified_generation != message.generation
                        ):
                            agreed = self.readiness_metadata(message.generation)
                            self._ready_notified_generation = message.generation
                            callback = self.on_generation_ready
                if callback is not None:
                    await callback(message.generation, *agreed)
        except (
            asyncio.IncompleteReadError,
            ConnectionError,
            BrokenPipeError,
            ProtocolError,
        ):
            pass
        finally:
            if worker_id is not None:
                async with self._lock:
                    if self._workers.get(worker_id) is connection:
                        self._workers.pop(worker_id, None)
                        self._prepared.pop(worker_id, None)
                        self._acknowledged.pop(worker_id, None)
            with contextlib.suppress(Exception):
                await connection.close()

    async def publish(self, message: MessageEnvelope) -> None:
        """Validate, retain, and fan out a master generation phase locally."""

        allowed = {"membership", "generation_rendezvous", "generation_active"}
        if message.message_type not in allowed:
            raise ValueError(f"local relay accepts only {sorted(allowed)} messages")
        async with self._lock:
            if message.message_type == "membership":
                self._latest_membership = message
                self._latest_rendezvous = None
                self._latest_active = None
                self._prepared.clear()
                self._acknowledged.clear()
                self._prepared_notified_generation = -1
                self._ready_notified_generation = -1
            elif message.message_type == "generation_rendezvous":
                latest = self._latest_membership
                if latest is None or message.generation != latest.generation:
                    raise ProtocolError("rendezvous does not match local membership")
                metadata = self.preparation_metadata(message.generation)
                if message.payload != {
                    "snapshot_hash": metadata[0],
                    "plan_checksum": metadata[1],
                    "compatibility_digest": metadata[2],
                }:
                    raise ProtocolError("rendezvous does not match local preparation")
                self._latest_rendezvous = message
            else:
                latest = self._latest_membership
                rendezvous = self._latest_rendezvous
                if latest is None or rendezvous is None or message.generation != latest.generation:
                    raise ProtocolError("active generation does not follow local rendezvous")
                metadata = self.readiness_metadata(message.generation)
                if message.payload != {
                    "snapshot_hash": metadata[0],
                    "plan_checksum": metadata[1],
                    "compatibility_digest": metadata[2],
                }:
                    raise ProtocolError("active generation does not match local readiness")
                self._latest_active = message
            workers = tuple(self._workers.items())
        results = await asyncio.gather(
            *(connection.send(message) for _, connection in workers),
            return_exceptions=True,
        )
        failed = [
            worker_id
            for (worker_id, _), result in zip(workers, results)
            if isinstance(result, BaseException)
        ]
        if failed:
            async with self._lock:
                for worker_id in failed:
                    self._workers.pop(worker_id, None)
                    self._prepared.pop(worker_id, None)
                    self._acknowledged.pop(worker_id, None)

    async def close(self) -> None:
        """Close workers and listener, then remove only the owned socket path."""

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        async with self._lock:
            workers = tuple(self._workers.values())
            self._workers.clear()
            self._prepared.clear()
            self._acknowledged.clear()
        await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)
        if self.path.exists() and stat.S_ISSOCK(self.path.stat().st_mode):
            self.path.unlink()


class LocalWorkerClient:
    """Execute the prepared/rendezvous/ready/active generation protocol."""

    def __init__(self, path: str | Path, node_id: str, worker_id: str) -> None:
        """Assign a fresh local incarnation and initialize phase cursors."""

        if not node_id or not worker_id:
            raise ValueError("node_id and worker_id are required")
        self.path = Path(path)
        self.node_id = node_id
        self.worker_id = worker_id
        self.incarnation_id = str(uuid.uuid4())
        self.connection: ControlConnection | None = None
        self.generation = -1
        self.active_generation = -1
        self.sequence = 0

    async def connect(self) -> None:
        """Open the Unix stream and register the stable worker identity."""

        reader, writer = await asyncio.open_unix_connection(self.path)
        self.connection = ControlConnection(reader, writer)
        await self.connection.send(
            MessageEnvelope(
                1,
                "worker_register",
                self.worker_id,
                self.incarnation_id,
                0,
                0,
                {"node_id": self.node_id},
            )
        )

    def _parse_membership(self, message: MessageEnvelope) -> MembershipSnapshot:
        """Verify and install a strictly newer checksummed membership."""

        if message.message_type != "membership":
            raise ProtocolError("local agent sent a non-membership message")
        snapshot = membership_snapshot_from_payload(message.generation, message.payload)
        if snapshot.generation <= self.generation:
            raise ProtocolError("local agent sent a stale membership generation")
        self.generation = snapshot.generation
        return snapshot

    async def receive(self) -> MembershipSnapshot:
        """Wait for membership while consuming informational phase messages."""

        if self.connection is None:
            raise RuntimeError("local worker is not connected")
        while True:
            message = await self.connection.receive()
            if message.message_type == "generation_active":
                if message.generation >= self.generation:
                    self.active_generation = message.generation
                continue
            if message.message_type == "generation_rendezvous":
                continue
            return self._parse_membership(message)

    @staticmethod
    def _metadata(execution_plan: Any, snapshot: MembershipSnapshot) -> tuple[str, str, str]:
        """Bind phase acknowledgements to membership, plan, and compatibility."""

        return (
            snapshot.snapshot_hash,
            str(getattr(execution_plan, "plan_checksum", snapshot.snapshot_hash)),
            str(getattr(execution_plan, "compatibility_digest", None) or ""),
        )

    async def _acknowledge(
        self,
        phase: str,
        snapshot: MembershipSnapshot,
        metadata: tuple[str, str, str],
    ) -> None:
        """Send one sequenced worker phase acknowledgement to the local relay."""

        assert self.connection is not None
        self.sequence += 1
        await self.connection.send(
            MessageEnvelope(
                1,
                "worker_ack",
                self.worker_id,
                self.incarnation_id,
                self.sequence,
                snapshot.generation,
                {
                    "phase": phase,
                    "snapshot_hash": metadata[0],
                    "plan_checksum": metadata[1],
                    "compatibility_digest": metadata[2],
                },
            )
        )

    @staticmethod
    def _validate_phase(
        message: MessageEnvelope,
        expected_type: str,
        snapshot: MembershipSnapshot,
        metadata: tuple[str, str, str],
    ) -> None:
        """Require the expected phase and exact locally prepared identities."""

        if message.message_type != expected_type:
            raise ProtocolError(
                f"local agent sent {message.message_type!r}; expected {expected_type}"
            )
        if message.generation != snapshot.generation or message.payload != {
            "snapshot_hash": metadata[0],
            "plan_checksum": metadata[1],
            "compatibility_digest": metadata[2],
        }:
            raise ProtocolError(f"{expected_type} does not match prepared membership")

    @staticmethod
    def _reprepare(prepared: Any, snapshot: MembershipSnapshot) -> Any:
        """Recompile ownership from a superseding snapshot without building WORLD."""

        owner = prepared.owner_plan
        tp = int(getattr(owner.parallel_config, "tensor_parallel_size"))
        if any(len(node.gpu_ids) != tp for node in snapshot.nodes):
            raise ValueError("membership does not match the fixed tensor-parallel width")
        coordinator = min(snapshot.nodes, key=lambda node: node.agent_id)
        owner.set_rendezvous_address(coordinator.addresses[0])
        owner.set_membership(tuple(node.agent_id for node in snapshot.nodes), snapshot.generation)
        return owner.prepare(
            prepared.device,
            prepared.dtype,
            recover_from_survivors=prepared.recover_from_survivors,
        )

    async def activate_prepared(self, prepared: Any, snapshot: MembershipSnapshot) -> Any:
        """Activate initial ownership only after both CPU control-plane barriers."""

        if self.connection is None:
            raise RuntimeError("local worker is not connected")
        if snapshot.generation != self.generation:
            raise ValueError("snapshot must be the latest received membership")
        while True:
            if prepared.execution_plan.generation != snapshot.generation:
                prepared = self._reprepare(prepared, snapshot)
            metadata = self._metadata(prepared.execution_plan, snapshot)
            await self._acknowledge("prepared", snapshot, metadata)
            while True:
                message = await self.connection.receive()
                if message.message_type == "membership":
                    snapshot = self._parse_membership(message)
                    prepared = self._reprepare(prepared, snapshot)
                    break
                self._validate_phase(message, "generation_rendezvous", snapshot, metadata)
                context = prepared.activate()
                await self._acknowledge("ready", snapshot, metadata)
                while True:
                    message = await self.connection.receive()
                    if message.message_type == "membership":
                        context.close()
                        snapshot = self._parse_membership(message)
                        prepared = self._reprepare(prepared, snapshot)
                        break
                    self._validate_phase(message, "generation_active", snapshot, metadata)
                    self.active_generation = message.generation
                    enable = getattr(context, "enable_control_plane_barrier", None)
                    if callable(enable):
                        enable()
                    mark = getattr(context, "mark_generation_active", None)
                    if callable(mark):
                        mark(message.generation)
                    return context
                break

    async def run_context(
        self,
        context: Any,
        *,
        initial_snapshot: MembershipSnapshot | None = None,
        on_snapshot: Callable[[MembershipSnapshot], Awaitable[None]] | None = None,
        on_active: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        """Drive reconfiguration barriers, restarting whenever membership supersedes."""

        if self.connection is None:
            raise RuntimeError("local worker is not connected")
        enable_barrier = getattr(context, "enable_control_plane_barrier", None)
        if callable(enable_barrier):
            enable_barrier()
        if initial_snapshot is not None and initial_snapshot.generation != self.generation:
            raise ValueError("initial snapshot must be the latest received membership")
        pending: MembershipSnapshot | None = initial_snapshot
        while True:
            snapshot = pending or await self.receive()
            pending = None
            context.apply_membership(snapshot)
            if on_snapshot is not None:
                await on_snapshot(snapshot)
            prepared_plan = getattr(context, "prepared_execution_plan", None)
            if prepared_plan is None:
                prepared_plan = getattr(context, "execution_plan", snapshot)
            metadata = self._metadata(prepared_plan, snapshot)
            await self._acknowledge("prepared", snapshot, metadata)
            while True:
                message = await self.connection.receive()
                if message.message_type == "membership":
                    pending = self._parse_membership(message)
                    break
                self._validate_phase(message, "generation_rendezvous", snapshot, metadata)
                activate = getattr(context, "activate_generation", None)
                if callable(activate):
                    activate()
                await self._acknowledge("ready", snapshot, metadata)
                while True:
                    message = await self.connection.receive()
                    if message.message_type == "membership":
                        pending = self._parse_membership(message)
                        break
                    self._validate_phase(message, "generation_active", snapshot, metadata)
                    self.active_generation = message.generation
                    mark_active = getattr(context, "mark_generation_active", None)
                    if callable(mark_active):
                        mark_active(message.generation)
                    if on_active is not None:
                        await on_active(message.generation)
                    break
                break

    async def close(self) -> None:
        """Close the worker-to-agent stream idempotently."""

        if self.connection is not None:
            await self.connection.close()
            self.connection = None


__all__ = ["LocalWorkerClient", "LocalWorkerRelay"]
