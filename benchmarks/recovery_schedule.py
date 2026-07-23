"""Machine-readable balanced-recovery scheduling benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from oobleck.state import LogicalStateEntry, StateManifest, plan_state_redistribution


def run(
    *, replicas: int, tensor_bytes: int, chunk_bytes: int, round_bytes: int | None = None
) -> dict[str, object]:
    if replicas < 1 or tensor_bytes < 1 or chunk_bytes < 1:
        raise ValueError("replicas and byte sizes must be positive")

    def entry(rank: int) -> LogicalStateEntry:
        return LogicalStateEntry(
            "layers.0.weight",
            (tensor_bytes,),
            (tensor_bytes,),
            "uint8",
            "parameter",
            ("replicate",),
            0,
            rank,
            1,
        )

    old = tuple(StateManifest(rank, (entry(rank),), 1) for rank in range(replicas))
    destination_rank = replicas
    new = (StateManifest(destination_rank, (entry(destination_rank),), 1),)
    started = time.perf_counter()
    schedule = plan_state_redistribution(
        old,
        new,
        chunk_bytes=chunk_bytes,
        round_bytes=round_bytes,
        alignment=min(256, chunk_bytes),
    )
    elapsed = time.perf_counter() - started
    balanced = dict(schedule.per_source_bytes)
    source_round_loads: dict[tuple[int, int], int] = {}
    destination_round_loads: dict[tuple[int, int], int] = {}
    for transfer in schedule.transfers:
        source_key = (transfer.round, transfer.source_rank)
        destination_key = (transfer.round, transfer.destination_rank)
        source_round_loads[source_key] = source_round_loads.get(source_key, 0) + transfer.byte_count
        destination_round_loads[destination_key] = (
            destination_round_loads.get(destination_key, 0) + transfer.byte_count
        )
    return {
        "schema_version": 1,
        "scenario": {
            "surviving_replicas": replicas,
            "destinations": 1,
            "tensor_bytes": tensor_bytes,
            "chunk_bytes": chunk_bytes,
            "round_bytes": round_bytes or chunk_bytes * replicas,
        },
        "schedule_hash": schedule.schedule_hash,
        "transfer_count": len(schedule.transfers),
        "balanced_source_bytes": balanced,
        "balanced_max_source_bytes": max(balanced.values(), default=0),
        "legacy_first_source_max_bytes": tensor_bytes,
        "max_source_load_reduction": 1.0 - max(balanced.values(), default=0) / tensor_bytes,
        "round_count": len({item.round for item in schedule.transfers}),
        "maximum_round_source_bytes": max(source_round_loads.values(), default=0),
        "maximum_round_destination_bytes": max(destination_round_loads.values(), default=0),
        "planning_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--tensor-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--round-bytes", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(
        replicas=arguments.replicas,
        tensor_bytes=arguments.tensor_bytes,
        chunk_bytes=arguments.chunk_bytes,
        round_bytes=arguments.round_bytes,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)


if __name__ == "__main__":
    main()
