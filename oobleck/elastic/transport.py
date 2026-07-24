"""Length-prefixed asyncio control transport with strict JSON envelopes."""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Mapping, Protocol


PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_BYTES = 1 << 20
_FIELDS = {
    "protocol_version",
    "message_type",
    "agent_id",
    "incarnation_id",
    "sequence_number",
    "generation",
    "payload",
}
_PAYLOAD_FIELDS = {
    "register": {"addresses", "gpu_ids"},
    "heartbeat": set(),
    "generation_prepared": {"snapshot_hash", "plan_checksum", "compatibility_digest"},
    "generation_ready": {"snapshot_hash", "plan_checksum", "compatibility_digest"},
    "drain": set(),
    "membership": {"nodes", "reasons", "snapshot_hash"},
    "generation_rendezvous": {"snapshot_hash", "plan_checksum", "compatibility_digest"},
    "generation_active": {"snapshot_hash", "plan_checksum", "compatibility_digest"},
    "inspect": set(),
    "inspect_status": set(),
    "status": {
        "nodes",
        "reasons",
        "snapshot_hash",
        "active_generation",
        "prepared_agents",
        "ready_agents",
    },
    "request_drain": {"node_id"},
    "drain_command": {"node_id"},
    "drain_accepted": {"node_id"},
    "worker_register": {"node_id"},
    "worker_ack": {"phase", "snapshot_hash", "plan_checksum", "compatibility_digest"},
}


class ProtocolError(ValueError):
    """A frame or envelope violates the versioned control protocol."""


class FrameTooLarge(ProtocolError):
    """A peer declared or encoded a frame beyond the configured bound."""


def _validate_payload(message_type: str, payload: Mapping[str, object]) -> None:
    """Enforce the exact field set and runtime types for each protocol message.

    Validation is intentionally closed-world: unknown message types, extra fields, bools
    masquerading as integers, malformed node inventories, or invalid phase metadata are
    rejected before service state machines see them. This keeps transport decoding from
    smuggling partially interpreted control state across generation boundaries.
    """

    expected = _PAYLOAD_FIELDS.get(message_type)
    if expected is None:
        raise ProtocolError(f"unsupported message_type {message_type!r}")
    if set(payload) != expected:
        raise ProtocolError(f"invalid {message_type} payload fields; expected={sorted(expected)}")
    if message_type == "register":
        for field in ("addresses", "gpu_ids"):
            values = payload[field]
            if (
                type(values) is not list
                or not values
                or not all(type(item) is str for item in values)
            ):
                raise ProtocolError(f"register {field} must be a non-empty list of strings")
    elif message_type in {"membership", "status"}:
        if type(payload["nodes"]) is not list or type(payload["reasons"]) is not list:
            raise ProtocolError(f"{message_type} nodes and reasons must be lists")
        if type(payload["snapshot_hash"]) is not str:
            raise ProtocolError(f"{message_type} snapshot_hash must be a string")
        if message_type == "status":
            if type(payload["active_generation"]) is not int or payload["active_generation"] < 0:
                raise ProtocolError("status active_generation must be non-negative")
            for field in ("prepared_agents", "ready_agents"):
                values = payload[field]
                if type(values) is not list or not all(type(item) is str for item in values):
                    raise ProtocolError(f"status {field} must be a list of strings")
    elif message_type in {
        "generation_prepared",
        "generation_ready",
        "generation_rendezvous",
        "generation_active",
    }:
        for field in ("snapshot_hash", "plan_checksum"):
            if type(payload[field]) is not str or not payload[field]:
                raise ProtocolError(f"{message_type} {field} must be a non-empty string")
        if type(payload["compatibility_digest"]) is not str:
            raise ProtocolError(f"{message_type} compatibility_digest must be a string")
    elif message_type in {
        "request_drain",
        "drain_command",
        "drain_accepted",
        "worker_register",
    }:
        if type(payload["node_id"]) is not str or not payload["node_id"]:
            raise ProtocolError(f"{message_type} node_id must be a non-empty string")
    elif message_type == "worker_ack":
        if payload["phase"] not in {"prepared", "ready"}:
            raise ProtocolError("worker_ack phase must be 'prepared' or 'ready'")
        for field in ("snapshot_hash", "plan_checksum"):
            if type(payload[field]) is not str or not payload[field]:
                raise ProtocolError(f"worker_ack {field} must be a non-empty string")
        if type(payload["compatibility_digest"]) is not str:
            raise ProtocolError("worker_ack compatibility_digest must be a string")


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    """Strictly validated metadata and payload for one control-plane message."""

    protocol_version: int
    message_type: str
    agent_id: str
    incarnation_id: str
    sequence_number: int
    generation: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        """Reject unknown versions, missing identities, and malformed payloads."""

        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version {self.protocol_version}; expected {PROTOCOL_VERSION}"
            )
        if not self.message_type or not self.agent_id or not self.incarnation_id:
            raise ProtocolError("message_type, agent_id, and incarnation_id are required")
        if self.sequence_number < 0 or self.generation < 0:
            raise ProtocolError("sequence_number and generation must be non-negative")
        if not isinstance(self.payload, Mapping):
            raise ProtocolError("payload must be a JSON object")
        _validate_payload(self.message_type, self.payload)

    @classmethod
    def from_dict(cls, value: object) -> "MessageEnvelope":
        """Decode only an exact envelope shape; unknown fields are protocol errors."""

        if not isinstance(value, dict):
            raise ProtocolError("control envelope must be a JSON object")
        if set(value) != _FIELDS:
            missing = sorted(_FIELDS - set(value))
            extra = sorted(set(value) - _FIELDS)
            raise ProtocolError(f"invalid envelope fields; missing={missing}, extra={extra}")
        expected_types = {
            "protocol_version": int,
            "message_type": str,
            "agent_id": str,
            "incarnation_id": str,
            "sequence_number": int,
            "generation": int,
            "payload": dict,
        }
        for name, expected in expected_types.items():
            if type(value[name]) is not expected:
                raise ProtocolError(f"{name} must be {expected.__name__}")
        return cls(**value)


def encode_frame(message: MessageEnvelope, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    """Serialize an envelope as bounded big-endian-length-prefixed JSON."""

    body = json.dumps(
        asdict(message), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if len(body) > max_frame_bytes:
        raise FrameTooLarge(f"control frame is {len(body)} bytes; limit is {max_frame_bytes}")
    return struct.pack(">I", len(body)) + body


async def read_frame(
    reader: asyncio.StreamReader, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> MessageEnvelope:
    """Read exactly one frame, preserving fragmentation and coalescing semantics."""

    header = await reader.readexactly(4)
    size = struct.unpack(">I", header)[0]
    if size == 0:
        raise ProtocolError("empty control frames are not allowed")
    if size > max_frame_bytes:
        raise FrameTooLarge(f"control frame declares {size} bytes; limit is {max_frame_bytes}")
    body = await reader.readexactly(size)
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("control frame is not valid UTF-8 JSON") from exc
    return MessageEnvelope.from_dict(value)


async def write_frame(
    writer: asyncio.StreamWriter,
    message: MessageEnvelope,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Write one complete frame and wait for stream backpressure."""

    writer.write(encode_frame(message, max_frame_bytes))
    await writer.drain()


class SerializedWriter:
    """One bounded writer queue per connection, providing real backpressure."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        queue_size: int = 64,
    ) -> None:
        """Start the sole writer task for this stream and bound queued sends."""

        self._writer = writer
        self._max_frame_bytes = max_frame_bytes
        self._queue: asyncio.Queue[tuple[MessageEnvelope | None, asyncio.Future[None]]] = (
            asyncio.Queue(queue_size)
        )
        self._task = asyncio.create_task(self._run())

    async def send(self, message: MessageEnvelope) -> None:
        """Queue a message and return only after it has drained to the stream."""

        if self._task.done():
            await self._task
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((message, future))
        await future

    async def _run(self) -> None:
        """Own the stream writer, preserving frame order and bounded backpressure.

        Exactly one task dequeues frames and awaits ``drain()``, preventing concurrent
        coroutines from interleaving bytes. A close sentinel flushes preceding messages.
        Any write failure is copied to its sender and every still-queued completion before
        the task terminates, so callers cannot mistake dropped control messages for success.
        """

        try:
            while True:
                message, completion = await self._queue.get()
                if message is None:
                    completion.set_result(None)
                    return
                try:
                    await write_frame(self._writer, message, self._max_frame_bytes)
                except BaseException as exc:
                    if not completion.done():
                        completion.set_exception(exc)
                    raise
                else:
                    completion.set_result(None)
        except BaseException as exc:
            while not self._queue.empty():
                _, completion = self._queue.get_nowait()
                if not completion.done():
                    completion.set_exception(exc)
            raise

    async def close(self) -> None:
        """Flush queued frames, stop the writer task, and close the stream."""

        if not self._task.done():
            completion = asyncio.get_running_loop().create_future()
            await self._queue.put((None, completion))
            await completion
            await self._task
        self._writer.close()
        await self._writer.wait_closed()


class ControlConnection:
    """Bidirectional framed connection with one serialized writer."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        writer_queue_size: int = 64,
    ) -> None:
        """Wrap the stream pair with shared frame and queue limits."""

        self.reader = reader
        self.writer = SerializedWriter(
            writer,
            max_frame_bytes=max_frame_bytes,
            queue_size=writer_queue_size,
        )
        self.max_frame_bytes = max_frame_bytes

    async def receive(self) -> MessageEnvelope:
        """Receive the next validated envelope from the peer."""

        return await read_frame(self.reader, self.max_frame_bytes)

    async def send(self, message: MessageEnvelope) -> None:
        """Send through the connection's ordered, backpressured queue."""

        await self.writer.send(message)

    async def close(self) -> None:
        """Close the serialized writer and its underlying stream."""

        await self.writer.close()


class ControlTransport(Protocol):
    """Transport boundary consumed by membership services and local IPC."""

    async def connect(self, host: str, port: int) -> ControlConnection:
        """Open one persistent client connection using the transport backend."""

        ...

    async def start_server(
        self,
        host: str,
        port: int,
        handler: Callable[[ControlConnection], Awaitable[None]],
    ) -> asyncio.AbstractServer:
        """Listen for framed connections and dispatch each to ``handler``."""

        ...


class AsyncioTcpControlTransport:
    """Asyncio stream implementation of the control transport contract."""

    def __init__(
        self,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        writer_queue_size: int = 64,
    ) -> None:
        """Configure identical frame and writer bounds for clients and servers."""

        if max_frame_bytes < 256 or writer_queue_size < 1:
            raise ValueError("invalid control transport limits")
        self.max_frame_bytes = max_frame_bytes
        self.writer_queue_size = writer_queue_size

    def _connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> ControlConnection:
        """Apply this transport's limits to a newly opened stream pair."""

        return ControlConnection(
            reader,
            writer,
            max_frame_bytes=self.max_frame_bytes,
            writer_queue_size=self.writer_queue_size,
        )

    async def connect(self, host: str, port: int) -> ControlConnection:
        """Open a persistent full-duplex TCP control connection."""

        reader, writer = await asyncio.open_connection(host, port)
        return self._connection(reader, writer)

    async def start_server(
        self,
        host: str,
        port: int,
        handler: Callable[[ControlConnection], Awaitable[None]],
    ) -> asyncio.AbstractServer:
        """Start a server that always closes each accepted connection after handling."""

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Adapt an asyncio stream pair and guarantee connection retirement."""

            connection = self._connection(reader, writer)
            try:
                await handler(connection)
            finally:
                await connection.close()

        return await asyncio.start_server(accept, host, port)
