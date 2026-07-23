from __future__ import annotations

import pytest

from oobleck.state import (
    LogicalStateEntry,
    StateManifest,
    StateUnavailable,
    plan_state_redistribution,
)


def entry(rank: int, logical_key: str, state_kind: str) -> LogicalStateEntry:
    return LogicalStateEntry(
        logical_key,
        (4,),
        (4,),
        "float32",
        state_kind,
        ("replicate",),
        0,
        rank,
        7,
    )


def manifest(rank: int, *, parameter: bool, optimizer: bool) -> StateManifest:
    entries = []
    if parameter:
        entries.append(entry(rank, "layer.weight", "parameter"))
    if optimizer:
        entries.append(entry(rank, "layer.weight::optimizer::exp_avg", "optimizer"))
    return StateManifest(rank, tuple(entries), 7)


def test_bundle_candidates_exclude_partial_optimizer_replicas():
    schedule = plan_state_redistribution(
        [
            manifest(0, parameter=True, optimizer=False),
            manifest(1, parameter=True, optimizer=True),
        ],
        [manifest(2, parameter=True, optimizer=True)],
        chunk_bytes=16,
        alignment=4,
    )
    assert {item.source_rank for item in schedule.transfers} == {1}
    assert dict(schedule.per_source_bytes) == {1: 32}


def test_bundle_without_one_complete_surviving_replica_is_rejected():
    with pytest.raises(StateUnavailable, match="complete state bundle"):
        plan_state_redistribution(
            [
                manifest(0, parameter=True, optimizer=False),
                manifest(1, parameter=False, optimizer=True),
            ],
            [manifest(2, parameter=True, optimizer=True)],
            chunk_bytes=16,
            alignment=4,
        )
