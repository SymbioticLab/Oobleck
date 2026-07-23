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
    "generation_ready": {"snapshot_hash"},
    "drain": set(),
    "membership": {"nodes", "reasons", "snapshot_hash"},
    "generation_active": {"snapshot_hash"},
    "inspect": set(),
    "request_drain": {"node_id"},
    "drain_command": {"node_id"},
    "drain_accepted": {"node_id"},
    "worker_register": {"node_id"},
    "worker_ack": {"phase"},
}


class ProtocolError(ValueError):
    pass


class FrameTooLarge(ProtocolError):
    pass


def _validate_payload(message_type: str, payload: Mapping[str, object]) -> None:
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
    elif message_type == "membership":
        if type(payload["nodes"]) is not list or type(payload["reasons"]) is not list:
            raise ProtocolError("membership nodes and reasons must be lists")
        if type(payload["snapshot_hash"]) is not str:
            raise ProtocolError("membership snapshot_hash must be a string")
    elif message_type in {"generation_ready", "generation_active"}:
        if type(payload["snapshot_hash"]) is not str or not payload["snapshot_hash"]:
            raise ProtocolError(f"{message_type} snapshot_hash must be a non-empty string")
    elif message_type in {
        "request_drain",
        "drain_command",
        "drain_accepted",
        "worker_register",
    }:
        if type(payload["node_id"]) is not str or not payload["node_id"]:
            raise ProtocolError(f"{message_type} node_id must be a non-empty string")
    elif message_type == "worker_ack" and payload["phase"] != "ready":
        raise ProtocolError("worker_ack phase must be 'ready'")


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    protocol_version: int
    message_type: str
    agent_id: str
    incarnation_id: str
    sequence_number: int
    generation: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
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
    body = json.dumps(
        asdict(message), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if len(body) > max_frame_bytes:
        raise FrameTooLarge(f"control frame is {len(body)} bytes; limit is {max_frame_bytes}")
    return struct.pack(">I", len(body)) + body


async def read_frame(
    reader: asyncio.StreamReader, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> MessageEnvelope:
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
        self._writer = writer
        self._max_frame_bytes = max_frame_bytes
        self._queue: asyncio.Queue[tuple[MessageEnvelope | None, asyncio.Future[None]]] = (
            asyncio.Queue(queue_size)
        )
        self._task = asyncio.create_task(self._run())

    async def send(self, message: MessageEnvelope) -> None:
        if self._task.done():
            await self._task
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((message, future))
        await future

    async def _run(self) -> None:
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
        if not self._task.done():
            completion = asyncio.get_running_loop().create_future()
            await self._queue.put((None, completion))
            await completion
            await self._task
        self._writer.close()
        await self._writer.wait_closed()


class ControlConnection:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        writer_queue_size: int = 64,
    ) -> None:
        self.reader = reader
        self.writer = SerializedWriter(
            writer,
            max_frame_bytes=max_frame_bytes,
            queue_size=writer_queue_size,
        )
        self.max_frame_bytes = max_frame_bytes

    async def receive(self) -> MessageEnvelope:
        return await read_frame(self.reader, self.max_frame_bytes)

    async def send(self, message: MessageEnvelope) -> None:
        await self.writer.send(message)

    async def close(self) -> None:
        await self.writer.close()


class ControlTransport(Protocol):
    async def connect(self, host: str, port: int) -> ControlConnection: ...

    async def start_server(
        self,
        host: str,
        port: int,
        handler: Callable[[ControlConnection], Awaitable[None]],
    ) -> asyncio.AbstractServer: ...


class AsyncioTcpControlTransport:
    def __init__(
        self,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        writer_queue_size: int = 64,
    ) -> None:
        if max_frame_bytes < 256 or writer_queue_size < 1:
            raise ValueError("invalid control transport limits")
        self.max_frame_bytes = max_frame_bytes
        self.writer_queue_size = writer_queue_size

    def _connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> ControlConnection:
        return ControlConnection(
            reader,
            writer,
            max_frame_bytes=self.max_frame_bytes,
            writer_queue_size=self.writer_queue_size,
        )

    async def connect(self, host: str, port: int) -> ControlConnection:
        reader, writer = await asyncio.open_connection(host, port)
        return self._connection(reader, writer)

    async def start_server(
        self,
        host: str,
        port: int,
        handler: Callable[[ControlConnection], Awaitable[None]],
    ) -> asyncio.AbstractServer:
        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            connection = self._connection(reader, writer)
            try:
                await handler(connection)
            finally:
                await connection.close()

        return await asyncio.start_server(accept, host, port)
