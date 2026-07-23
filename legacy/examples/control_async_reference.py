from __future__ import annotations

import asyncio

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


@pytest.mark.asyncio
async def test_frames_support_fragmentation_and_coalescing():
    first = MessageEnvelope(1, "heartbeat", "a", "i", 1, 0, {})
    second = MessageEnvelope(1, "heartbeat", "a", "i", 2, 0, {})
    payload = encode_frame(first) + encode_frame(second)
    reader = asyncio.StreamReader()
    for byte in payload[:7]:
        reader.feed_data(bytes([byte]))
    reader.feed_data(payload[7:])
    reader.feed_eof()
    assert await read_frame(reader) == first
    assert await read_frame(reader) == second


def test_oversized_frame_is_rejected():
    message = MessageEnvelope(1, "register", "a", "i", 0, 0, {"value": "x" * 100})
    with pytest.raises(FrameTooLarge):
        encode_frame(message, max_frame_bytes=32)


def test_membership_coalesces_failures_and_rejects_stale_messages():
    now = 0.0
    state = MembershipStateMachine(lease_timeout_s=5, clock=lambda: now)
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
