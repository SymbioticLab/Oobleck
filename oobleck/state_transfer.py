"""Collective execution of immutable state transfer schedules."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from oobleck.state import TransferSchedule, all_to_all_single_compat

StateTensorKey = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class TransferExecutionMetrics:
    actual_source_bytes: tuple[tuple[int, int], ...]
    actual_destination_bytes: tuple[tuple[int, int], ...]
    actual_link_class_bytes: tuple[tuple[str, int], ...]
    round_durations: tuple[tuple[int, str, float], ...]


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError("state transfer tensors must be contiguous")
    # ``view(dtype)`` rejects zero-dimensional tensors when element sizes
    # differ.  Normalizing to a one-dimensional storage view also covers
    # scalar optimizer slots such as Adam's step counter.
    return tensor.reshape(-1).view(torch.uint8).flatten()


def _transfer_identity(item) -> tuple[object, ...]:
    return (
        item.round,
        item.source_rank,
        item.destination_rank,
        item.logical_key,
        item.state_kind,
        item.tp_lane,
        item.chunk_start,
        item.chunk_end,
        item.version,
    )


def _checksum(payload: torch.Tensor) -> str:
    return hashlib.sha256(payload.detach().cpu().numpy().tobytes()).hexdigest()


def _validate_dtype(tensor: torch.Tensor, expected: str, key: StateTensorKey) -> None:
    actual = str(tensor.dtype).removeprefix("torch.")
    if actual != expected.removeprefix("torch."):
        raise ValueError(
            f"state transfer dtype mismatch for {key}: expected {expected}, got {actual}"
        )


def execute_transfer_schedule(
    schedule: TransferSchedule,
    *,
    rank: int,
    world_size: int,
    sources: Mapping[StateTensorKey, torch.Tensor],
    destinations: MutableMapping[StateTensorKey, torch.Tensor],
    device: torch.device | str,
    group=None,
    verify_checksums: bool = True,
) -> TransferExecutionMetrics:
    """Pack, all-to-all, validate, and unpack every collective round.

    Each rank calls this with the same schedule and participates with zero-sized
    splits when it has no payload. Destination tensors are preallocated from the
    new manifest, allowing unpacking directly into their storage.
    """

    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world_size")
    target_device = torch.device(device)
    round_durations = []
    rounds = sorted({(item.round, item.dtype) for item in schedule.transfers})
    for round_number, dtype in rounds:
        round_started = time.perf_counter()
        selected = [
            item
            for item in schedule.transfers
            if item.round == round_number and item.dtype == dtype
        ]
        outgoing = [item for item in selected if item.source_rank == rank]
        incoming = [item for item in selected if item.destination_rank == rank]
        outgoing.sort(
            key=lambda item: (
                item.destination_rank,
                item.logical_key,
                item.state_kind,
                item.tp_lane,
                item.chunk_start,
            )
        )
        incoming.sort(
            key=lambda item: (
                item.source_rank,
                item.logical_key,
                item.state_kind,
                item.tp_lane,
                item.chunk_start,
            )
        )
        input_splits = [0] * world_size
        output_splits = [0] * world_size
        packed = []
        local_checksums = {}
        for item in outgoing:
            key = (item.logical_key, item.state_kind, item.tp_lane)
            if key not in sources:
                raise KeyError(f"source rank {rank} is missing scheduled state {key}")
            _validate_dtype(sources[key], item.dtype, key)
            payload = _bytes(sources[key])
            if item.chunk_end > payload.numel():
                raise ValueError(f"scheduled chunk exceeds source tensor for {key}")
            chunk = payload[item.chunk_start : item.chunk_end]
            packed.append(chunk.to(target_device))
            if verify_checksums:
                local_checksums[_transfer_identity(item)] = _checksum(chunk)
            input_splits[item.destination_rank] += item.byte_count
        for item in incoming:
            output_splits[item.source_rank] += item.byte_count
        input_buffer = (
            torch.cat(packed) if packed else torch.empty(0, dtype=torch.uint8, device=target_device)
        )
        output_buffer = torch.empty(sum(output_splits), dtype=torch.uint8, device=target_device)
        expected_checksums = {}
        if verify_checksums:
            import torch.distributed as dist

            checksum_maps: list[dict[tuple[object, ...], str] | None] = [None] * world_size
            dist.all_gather_object(checksum_maps, local_checksums, group=group)
            for values in checksum_maps:
                if values is not None:
                    expected_checksums.update(values)
        all_to_all_single_compat(
            output_buffer,
            input_buffer,
            output_splits,
            input_splits,
            group=group,
        )
        cursor = 0
        for item in incoming:
            key = (item.logical_key, item.state_kind, item.tp_lane)
            if key not in destinations:
                raise KeyError(f"destination rank {rank} is missing allocated state {key}")
            _validate_dtype(destinations[key], item.dtype, key)
            payload = _bytes(destinations[key])
            if item.chunk_end > payload.numel():
                raise ValueError(f"scheduled chunk exceeds destination tensor for {key}")
            end = cursor + item.byte_count
            payload[item.chunk_start : item.chunk_end].copy_(output_buffer[cursor:end])
            if verify_checksums:
                identity = _transfer_identity(item)
                expected = expected_checksums.get(identity)
                actual = _checksum(payload[item.chunk_start : item.chunk_end])
                if expected is None or actual != expected:
                    raise RuntimeError(
                        f"state transfer checksum mismatch for {key} "
                        f"bytes [{item.chunk_start}, {item.chunk_end})"
                    )
            cursor = end
        if cursor != output_buffer.numel():
            raise RuntimeError("state transfer unpack byte count mismatch")
        elapsed = time.perf_counter() - round_started
        if world_size > 1:
            import torch.distributed as dist

            duration_values: list[float | None] = [None] * world_size
            dist.all_gather_object(duration_values, elapsed, group=group)
            elapsed = max(value for value in duration_values if value is not None)
        round_durations.append((round_number, dtype, elapsed))
    return TransferExecutionMetrics(
        schedule.per_source_bytes,
        schedule.per_destination_bytes,
        schedule.per_link_class_bytes,
        tuple(round_durations),
    )
