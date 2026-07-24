"""Stable logical state manifests and balanced all-to-all transfer planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import reduce
from operator import mul
from typing import Callable, Iterable, Mapping, Sequence


class StateUnavailable(RuntimeError):
    """No surviving source owns a required committed state shard."""


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _dtype_name(value: str) -> str:
    """Normalize torch-qualified dtype names for stable manifests and grouping."""

    return value.removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class LogicalStateEntry:
    """Versioned ownership metadata for one logical parameter, buffer, or slot."""

    logical_key: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    state_kind: str
    placements: tuple[str, ...]
    tp_lane: int
    owner_rank: int
    version: int
    global_layer_id: str | None = None
    shared_state_id: str | None = None
    byte_count: int | None = None

    def __post_init__(self) -> None:
        """Validate shard identity and ensure its storage size is computable."""

        if not self.logical_key:
            raise ValueError("logical_key must not be empty")
        if any(size < 0 for size in (*self.global_shape, *self.local_shape)):
            raise ValueError("state shapes must be non-negative")
        if self.tp_lane < 0 or self.owner_rank < 0 or self.version < 0:
            raise ValueError("tp_lane, owner_rank, and version must be non-negative")
        if self.byte_count is not None and self.byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if self.byte_count is None and _dtype_name(self.dtype) not in _DTYPE_BYTES:
            raise ValueError(f"unknown dtype {self.dtype!r}; provide byte_count explicitly")

    @property
    def nbytes(self) -> int:
        """Return explicit storage bytes or derive them from local shape and dtype."""

        if self.byte_count is not None:
            return self.byte_count
        elements = reduce(mul, self.local_shape, 1)
        return elements * _DTYPE_BYTES[_dtype_name(self.dtype)]

    @property
    def shard_identity(self) -> tuple[object, ...]:
        """Return the fields that must match for a committed shard to be reused."""

        return (
            self.logical_key,
            self.tp_lane,
            self.local_shape,
            _dtype_name(self.dtype),
            self.placements,
            self.version,
        )


@dataclass(frozen=True, slots=True)
class StateManifest:
    """Checksummed committed-state inventory owned by one rank."""

    rank: int
    entries: tuple[LogicalStateEntry, ...]
    committed_step: int
    manifest_hash: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Validate ownership uniqueness and compute the consensus hash."""

        if self.rank < 0 or self.committed_step < 0:
            raise ValueError("rank and committed_step must be non-negative")
        if any(entry.owner_rank != self.rank for entry in self.entries):
            raise ValueError("manifest entries must be owned by the manifest rank")
        keys = [(entry.logical_key, entry.tp_lane, entry.state_kind) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("manifest contains duplicate logical shards")
        payload = {
            "rank": self.rank,
            "committed_step": self.committed_step,
            "entries": [asdict(item) for item in self.entries],
        }
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("manifest hash is invalid")
        object.__setattr__(self, "manifest_hash", expected)


@dataclass(frozen=True, slots=True)
class Transfer:
    """One byte range assigned to a source, destination, dtype, and round."""

    source_rank: int
    destination_rank: int
    logical_key: str
    state_kind: str
    tp_lane: int
    chunk_start: int
    chunk_end: int
    dtype: str
    round: int
    link_class: str
    version: int

    @property
    def byte_count(self) -> int:
        """Return the half-open chunk length in bytes."""

        return self.chunk_end - self.chunk_start


@dataclass(frozen=True, slots=True)
class TransferSchedule:
    """Immutable redistribution plan with deterministic load accounting."""

    transfers: tuple[Transfer, ...]
    retained: tuple[tuple[int, str, int], ...]
    per_source_bytes: tuple[tuple[int, int], ...]
    per_destination_bytes: tuple[tuple[int, int], ...]
    per_link_class_bytes: tuple[tuple[str, int], ...]
    schedule_hash: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Compute or verify the schedule hash exchanged by all ranks."""

        payload = {
            "transfers": [asdict(item) for item in self.transfers],
            "retained": self.retained,
        }
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.schedule_hash and self.schedule_hash != expected:
            raise ValueError("transfer schedule hash is invalid")
        object.__setattr__(self, "schedule_hash", expected)

    def split_sizes(
        self, rank: int, world_size: int, *, round: int = 0, dtype: str | None = None
    ) -> tuple[list[int], list[int]]:
        """Build all-to-all input/output byte splits for one rank and round."""

        inputs = [0] * world_size
        outputs = [0] * world_size
        for item in self.transfers:
            if item.round != round or (dtype is not None and item.dtype != dtype):
                continue
            if item.source_rank == rank:
                inputs[item.destination_rank] += item.byte_count
            if item.destination_rank == rank:
                outputs[item.source_rank] += item.byte_count
        return inputs, outputs


def _chunks(size: int, chunk_bytes: int, alignment: int) -> Iterable[tuple[int, int]]:
    """Yield aligned half-open byte ranges without padding the final chunk."""

    if size == 0:
        return
    effective = max(alignment, (chunk_bytes // alignment) * alignment)
    start = 0
    while start < size:
        end = min(size, start + effective)
        yield start, end
        start = end


def _bundle_key(entry: LogicalStateEntry) -> tuple[str, int, int]:
    """Keep a parameter and all of its optimizer slots on one complete source."""

    parameter_key = entry.logical_key.split("::optimizer::", 1)[0]
    return parameter_key, entry.tp_lane, entry.version


def plan_state_redistribution(
    old_manifests: Sequence[StateManifest],
    new_manifests: Sequence[StateManifest],
    *,
    chunk_bytes: int,
    round_bytes: int | None = None,
    alignment: int = 256,
    rank_to_node: Mapping[int, str] | None = None,
    bandwidth_bytes_per_s: Mapping[int, float] | None = None,
    link_classifier: Callable[[int, int], tuple[str, float]] | None = None,
) -> TransferSchedule:
    """Assign largest chunks to candidates by projected byte makespan."""

    if chunk_bytes < 1 or alignment < 1:
        raise ValueError("chunk_bytes and alignment must be positive")
    round_bytes = round_bytes or chunk_bytes * max(1, len(old_manifests))
    if round_bytes < chunk_bytes:
        raise ValueError("round_bytes must be at least chunk_bytes")
    old_entries = [entry for manifest in old_manifests for entry in manifest.entries]
    candidates: dict[tuple[object, ...], list[LogicalStateEntry]] = {}
    for entry in old_entries:
        candidates.setdefault(entry.shard_identity, []).append(entry)
    source_load: dict[int, int] = {}
    destination_load: dict[int, int] = {}
    link_load: dict[str, int] = {}
    retained: list[tuple[int, str, int]] = []
    units: list[tuple[int, LogicalStateEntry, int, int, tuple[LogicalStateEntry, ...]]] = []

    for manifest in sorted(new_manifests, key=lambda item: item.rank):
        destinations = tuple(
            sorted(
                manifest.entries,
                key=lambda item: (item.logical_key, item.tp_lane, item.state_kind),
            )
        )
        versions = {entry.version for entry in destinations}
        if versions and versions != {manifest.committed_step}:
            raise ValueError("destination manifest mixes committed state versions")

        bundles: dict[tuple[str, int, int], list[LogicalStateEntry]] = {}
        for destination in destinations:
            bundles.setdefault(_bundle_key(destination), []).append(destination)
        complete_sources: dict[tuple[str, int, int], set[int]] = {}
        for bundle, entries in sorted(bundles.items()):
            rank_sets = []
            for destination in entries:
                ranks = {
                    entry.owner_rank for entry in candidates.get(destination.shard_identity, ())
                }
                if not ranks:
                    raise StateUnavailable(
                        f"no committed source for {destination.logical_key} "
                        f"(tp lane {destination.tp_lane}, version {destination.version})"
                    )
                rank_sets.append(ranks)
            complete = set.intersection(*rank_sets)
            if not complete:
                raise StateUnavailable(
                    f"no surviving rank holds the complete state bundle {bundle[0]} "
                    f"(tp lane {bundle[1]}, version {bundle[2]})"
                )
            complete_sources[bundle] = complete

        for destination in destinations:
            complete = complete_sources[_bundle_key(destination)]
            eligible = tuple(
                sorted(
                    (
                        entry
                        for entry in candidates[destination.shard_identity]
                        if entry.owner_rank in complete
                    ),
                    key=lambda item: item.owner_rank,
                )
            )
            if destination.owner_rank in complete:
                retained.append(
                    (destination.owner_rank, destination.logical_key, destination.tp_lane)
                )
                continue
            for start, end in _chunks(destination.nbytes, chunk_bytes, alignment):
                units.append((end - start, destination, start, end, eligible))

    units.sort(
        key=lambda item: (
            -item[0],
            item[1].logical_key,
            item[1].tp_lane,
            item[1].owner_rank,
            item[2],
        )
    )
    transfers: list[Transfer] = []
    round_source_loads: list[dict[int, int]] = []
    round_destination_loads: list[dict[int, int]] = []

    def locality(source: int, destination: int) -> tuple[str, float]:
        """Classify a candidate link and return its relative transfer cost."""

        if link_classifier is not None:
            return link_classifier(source, destination)
        if rank_to_node and rank_to_node.get(source) == rank_to_node.get(destination):
            return "same-node", 0.25
        return "network", 1.0

    for size, destination, start, end, eligible in units:
        scored = []
        for source in eligible:
            link_class, locality_cost = locality(source.owner_rank, destination.owner_rank)
            source_bw = (bandwidth_bytes_per_s or {}).get(source.owner_rank, 1.0)
            destination_bw = (bandwidth_bytes_per_s or {}).get(destination.owner_rank, 1.0)
            projected = max(
                (source_load.get(source.owner_rank, 0) + size) / source_bw,
                (destination_load.get(destination.owner_rank, 0) + size) / destination_bw,
                (link_load.get(link_class, 0) + size) * locality_cost,
            )
            scored.append((projected, locality_cost, source.owner_rank, source, link_class))
        _, _, _, source, link_class = min(scored, key=lambda item: item[:3])
        round_number = 0
        while round_number < len(round_source_loads):
            if (
                round_source_loads[round_number].get(source.owner_rank, 0) + size <= round_bytes
                and round_destination_loads[round_number].get(destination.owner_rank, 0) + size
                <= round_bytes
            ):
                break
            round_number += 1
        if round_number == len(round_source_loads):
            round_source_loads.append({})
            round_destination_loads.append({})
        round_source_loads[round_number][source.owner_rank] = (
            round_source_loads[round_number].get(source.owner_rank, 0) + size
        )
        round_destination_loads[round_number][destination.owner_rank] = (
            round_destination_loads[round_number].get(destination.owner_rank, 0) + size
        )
        source_load[source.owner_rank] = source_load.get(source.owner_rank, 0) + size
        destination_load[destination.owner_rank] = (
            destination_load.get(destination.owner_rank, 0) + size
        )
        link_load[link_class] = link_load.get(link_class, 0) + size
        transfers.append(
            Transfer(
                source.owner_rank,
                destination.owner_rank,
                destination.logical_key,
                destination.state_kind,
                destination.tp_lane,
                start,
                end,
                _dtype_name(destination.dtype),
                round_number,
                link_class,
                destination.version,
            )
        )

    transfers.sort(
        key=lambda item: (
            item.round,
            item.dtype,
            item.source_rank,
            item.destination_rank,
            item.logical_key,
            item.tp_lane,
            item.chunk_start,
        )
    )
    return TransferSchedule(
        tuple(transfers),
        tuple(sorted(retained)),
        tuple(sorted(source_load.items())),
        tuple(sorted(destination_load.items())),
        tuple(sorted(link_load.items())),
    )


def all_to_all_single_compat(
    output,
    input,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    *,
    group=None,
    async_op: bool = False,
):
    """Version-isolated wrapper around PyTorch's experimental collective."""

    import torch.distributed as dist

    return dist.all_to_all_single(
        output,
        input,
        output_split_sizes=list(output_split_sizes),
        input_split_sizes=list(input_split_sizes),
        group=group,
        async_op=async_op,
    )
