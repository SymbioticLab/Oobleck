from __future__ import annotations

import asyncio
import json
import struct

import pytest

from oobleck.elastic import (
    FrameTooLarge,
    MembershipStateMachine,
    MessageEnvelope,
    NodeIdentity,
    StaleGeneration,
    StaleSequence,
    encode_frame,
    read_frame,
)
from oobleck.elastic.transport import ProtocolError, SerializedWriter


def test_frames_support_fragmentation_and_coalescing():
    async def check():
        first = MessageEnvelope("heartbeat", "a", "i", 1, 0, {})
        second = MessageEnvelope("heartbeat", "a", "i", 2, 0, {})
        first_frame = encode_frame(first)
        assert b"protocol_version" not in first_frame
        payload = first_frame + encode_frame(second)
        reader = asyncio.StreamReader()
        for byte in payload[:7]:
            reader.feed_data(bytes([byte]))
        reader.feed_data(payload[7:])
        reader.feed_eof()
        assert await read_frame(reader) == first
        assert await read_frame(reader) == second

    asyncio.run(check())


def test_oversized_frame_is_rejected():
    message = MessageEnvelope(
        "register",
        "a",
        "i",
        0,
        0,
        {"addresses": ["x" * 100], "gpu_ids": ["0"]},
    )
    with pytest.raises(FrameTooLarge):
        encode_frame(message, max_frame_bytes=32)
    with pytest.raises(ProtocolError, match="non-empty list of strings"):
        MessageEnvelope("register", "a", "i", 0, 0, {"addresses": "host", "gpu_ids": [0]})


def test_malformed_frame_and_strict_field_types_are_rejected():
    async def check():
        reader = asyncio.StreamReader()
        body = json.dumps({"message_type": True}).encode()
        reader.feed_data(struct.pack(">I", len(body)) + body)
        with pytest.raises(ProtocolError):
            await read_frame(reader)

    asyncio.run(check())


def test_serialized_writer_applies_drain_backpressure():
    class Writer:
        def __init__(self):
            self.release = asyncio.Event()
            self.closed = False

        def write(self, value):
            self.value = value

        async def drain(self):
            await self.release.wait()

        def close(self):
            self.closed = True

        async def wait_closed(self):
            pass

    async def check():
        raw = Writer()
        writer = SerializedWriter(raw, queue_size=1)
        message = MessageEnvelope("heartbeat", "a", "i", 1, 0, {})
        pending = asyncio.create_task(writer.send(message))
        await asyncio.sleep(0)
        assert not pending.done()
        raw.release.set()
        await pending
        await writer.close()
        assert raw.closed

    asyncio.run(check())


def test_membership_coalesces_failures_and_rejects_stale_messages():
    state = MembershipStateMachine(lease_timeout_s=5, clock=lambda: 0.0)
    a = NodeIdentity("a", "a1", ("10.0.0.1",), ("0",))
    b = NodeIdentity("b", "b1", ("10.0.0.2",), ("0",))
    state.register(a, 0)
    state.register(b, 0)
    initial = state.publish()
    assert initial is not None and initial.generation == 1
    with pytest.raises(StaleGeneration):
        state.heartbeat("a", "a1", 1, 0)
    state.heartbeat("a", "a1", 1, 1)
    with pytest.raises(StaleSequence):
        state.heartbeat("a", "a1", 1, 1)
    state.disconnect("a", "a1")
    state.disconnect("b", "b1")
    failed = state.publish()
    assert failed is not None and failed.generation == 2
    assert failed.nodes == ()
    assert len(failed.reasons) == 2


def test_incomplete_frame_and_lease_expiry_use_failure_path():
    async def incomplete():
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 8) + b"{}")
        reader.feed_eof()
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(reader)

    asyncio.run(incomplete())

    now = [0.0]
    state = MembershipStateMachine(lease_timeout_s=5, clock=lambda: now[0])
    state.register(NodeIdentity("a", "a1", ("10.0.0.1",), ("0",)), 0)
    assert state.publish().generation == 1
    now[0] = 5.0
    assert state.expire_leases() == ("a",)
    expired = state.publish()
    assert expired is not None and expired.generation == 2
    assert expired.nodes == ()
    assert expired.reasons == ("lease-expired:a",)
