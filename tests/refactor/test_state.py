import pytest
import torch

from oobleck.state import (
    LogicalStateEntry,
    StateManifest,
    StateUnavailable,
    Transfer,
    TransferSchedule,
    plan_state_redistribution,
)
from oobleck.state_transfer import execute_transfer_schedule


def entry(rank: int, *, version: int = 3, size: int = 16):
    return LogicalStateEntry(
        "layers.0.weight",
        (size,),
        (size,),
        "float32",
        "parameter",
        ("shard:0",),
        0,
        rank,
        version,
    )


def test_large_tensor_is_striped_by_bytes_across_replicas():
    schedule = plan_state_redistribution(
        [StateManifest(0, (entry(0),), 3), StateManifest(1, (entry(1),), 3)],
        [StateManifest(2, (entry(2),), 3)],
        chunk_bytes=16,
        round_bytes=16,
        alignment=4,
    )
    assert dict(schedule.per_source_bytes) == {0: 32, 1: 32}
    assert sum(item.byte_count for item in schedule.transfers) == 64
    for round_number in {item.round for item in schedule.transfers}:
        selected = [item for item in schedule.transfers if item.round == round_number]
        for source in {item.source_rank for item in selected}:
            assert sum(item.byte_count for item in selected if item.source_rank == source) <= 16
        for destination in {item.destination_rank for item in selected}:
            assert (
                sum(item.byte_count for item in selected if item.destination_rank == destination)
                <= 16
            )
    assert schedule.schedule_hash


def test_retention_and_stale_version_filtering():
    retained = plan_state_redistribution(
        [StateManifest(0, (entry(0),), 3)],
        [StateManifest(0, (entry(0),), 3)],
        chunk_bytes=16,
    )
    assert retained.transfers == ()
    assert retained.retained == ((0, "layers.0.weight", 0),)
    with pytest.raises(StateUnavailable, match="no committed source"):
        plan_state_redistribution(
            [StateManifest(0, (entry(0, version=2),), 2)],
            [StateManifest(1, (entry(1, version=3),), 3)],
            chunk_bytes=16,
        )


def test_transfer_schedule_hash_and_tensor_dtype_are_validated_before_collectives():
    transfer = Transfer(
        0,
        1,
        "layers.0.weight",
        "parameter",
        0,
        0,
        4,
        "float32",
        0,
        "network",
        3,
    )
    schedule = TransferSchedule((transfer,), (), ((0, 4),), ((1, 4),), (("network", 4),))
    with pytest.raises(ValueError, match="schedule hash"):
        TransferSchedule(
            schedule.transfers,
            schedule.retained,
            schedule.per_source_bytes,
            schedule.per_destination_bytes,
            schedule.per_link_class_bytes,
            "invalid",
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        execute_transfer_schedule(
            schedule,
            rank=0,
            world_size=2,
            sources={("layers.0.weight", "parameter", 0): torch.ones(4, dtype=torch.uint8)},
            destinations={},
            device="cpu",
            verify_checksums=False,
        )


def test_locality_precedes_stable_rank_and_dtype_buckets_have_exact_splits():
    local = plan_state_redistribution(
        [StateManifest(0, (entry(0),), 3), StateManifest(1, (entry(1),), 3)],
        [StateManifest(2, (entry(2),), 3)],
        chunk_bytes=64,
        rank_to_node={0: "node-a", 1: "node-b", 2: "node-a"},
    )
    assert {item.source_rank for item in local.transfers} == {0}
    assert {item.link_class for item in local.transfers} == {"same-node"}

    def integer_entry(rank):
        return LogicalStateEntry(
            "optimizer.step",
            (2,),
            (2,),
            "int64",
            "buffer",
            ("replicate",),
            0,
            rank,
            3,
        )

    source = StateManifest(0, (entry(0), integer_entry(0)), 3)
    destination = StateManifest(1, (entry(1), integer_entry(1)), 3)
    bucketed = plan_state_redistribution([source], [destination], chunk_bytes=64)
    assert {item.dtype for item in bucketed.transfers} == {"float32", "int64"}
    float_round = next(item.round for item in bucketed.transfers if item.dtype == "float32")
    int_round = next(item.round for item in bucketed.transfers if item.dtype == "int64")
    float_inputs, _ = bucketed.split_sizes(0, 2, round=float_round, dtype="float32")
    int_inputs, _ = bucketed.split_sizes(0, 2, round=int_round, dtype="int64")
    assert float_inputs == [0, 64]
    assert int_inputs == [0, 16]
