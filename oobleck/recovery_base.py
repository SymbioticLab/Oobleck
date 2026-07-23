"""Committed training-state capture and deterministic generation recovery."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping, MutableMapping

import torch

from oobleck.optimization import (
    OptimizerStateSchema,
    optimizer_schema_for_partition,
    restore_optimizer_state,
    serialize_optimizer_state,
)
from oobleck.state import StateManifest, TransferSchedule, plan_state_redistribution
from oobleck.state_transfer import StateTensorKey, execute_transfer_schedule


@dataclass(slots=True)
class RecoverySnapshot:
    manifest: StateManifest
    tensors: dict[StateTensorKey, torch.Tensor]
    optimizer_schema: OptimizerStateSchema | None
    scheduler_state: Mapping[str, Any] | None
    scaler_state: Mapping[str, Any] | None
    committed_step: int


@dataclass(frozen=True, slots=True)
class RecoveryMetadata:
    manifest: StateManifest
    optimizer_schema: OptimizerStateSchema | None
    scheduler_state: Mapping[str, Any] | None
    scaler_state: Mapping[str, Any] | None
    committed_step: int


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    schedule: TransferSchedule
    source_bytes: tuple[tuple[int, int], ...]
    destination_bytes: tuple[tuple[int, int], ...]
    link_class_bytes: tuple[tuple[str, int], ...]
    actual_source_bytes: tuple[tuple[int, int], ...] = ()
    actual_destination_bytes: tuple[tuple[int, int], ...] = ()
    actual_link_class_bytes: tuple[tuple[str, int], ...] = ()
    round_durations: tuple[tuple[int, str, float], ...] = ()
    metadata_exchange_seconds: float = 0.0
    planning_seconds: float = 0.0
    transfer_seconds: float = 0.0
    restoration_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def maximum_source_bytes(self) -> int:
        return max((value for _, value in self.source_bytes), default=0)

    @property
    def maximum_destination_bytes(self) -> int:
        return max((value for _, value in self.destination_bytes), default=0)

    @property
    def source_scheduling_error_bytes(self) -> int:
        planned = dict(self.source_bytes)
        actual = dict(self.actual_source_bytes)
        ranks = set(planned) | set(actual)
        return max((abs(actual.get(rank, 0) - planned.get(rank, 0)) for rank in ranks), default=0)

    @property
    def straggler_round_seconds(self) -> float:
        return max((duration for _, _, duration in self.round_durations), default=0.0)


def version_manifest(manifest: StateManifest, committed_step: int) -> StateManifest:
    return StateManifest(
        manifest.rank,
        tuple(replace(entry, version=committed_step) for entry in manifest.entries),
        committed_step,
    )


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _logical_candidates(root: torch.nn.Module, local_name: str) -> tuple[str, ...]:
    candidates = [local_name]
    modules = sorted(root.named_modules(), key=lambda item: len(item[0]), reverse=True)
    for prefix, module in modules:
        converter = getattr(module, "_global_cornstarch_key", None)
        if not callable(converter):
            continue
        if prefix and not local_name.startswith(f"{prefix}."):
            continue
        relative = local_name[len(prefix) + 1 :] if prefix else local_name
        try:
            global_name = converter(relative)
        except (KeyError, ValueError, AttributeError):
            continue
        candidates.append(str(global_name))
        if prefix:
            candidates.append(f"{prefix}.{global_name}")
    return tuple(dict.fromkeys(candidates))


def logical_state_bindings(
    model: torch.nn.Module,
    manifest: StateManifest,
) -> tuple[dict[str, torch.nn.Parameter], dict[StateTensorKey, torch.Tensor]]:
    """Bind stable manifest identities to the active partition's local tensors."""

    parameters: dict[str, torch.nn.Parameter] = {}
    states: dict[StateTensorKey, torch.Tensor] = {}
    expected = {(entry.logical_key, entry.state_kind): entry for entry in manifest.entries}
    for kind, values in (
        ("parameter", model.named_parameters(recurse=True, remove_duplicate=False)),
        ("buffer", model.named_buffers(recurse=True, remove_duplicate=False)),
    ):
        for local_name, value in values:
            entry = None
            for candidate in _logical_candidates(model, local_name):
                entry = expected.get((candidate, kind))
                if entry is not None:
                    break
            if entry is None:
                continue
            local = _local_tensor(value)
            if tuple(local.shape) != entry.local_shape or str(local.dtype) != entry.dtype:
                raise RuntimeError(
                    f"active tensor {entry.logical_key!r} does not match its manifest"
                )
            key = (entry.logical_key, kind, entry.tp_lane)
            states[key] = local
            if kind == "parameter":
                parameters[entry.logical_key] = value

    missing = [
        (entry.logical_key, entry.state_kind, entry.tp_lane)
        for entry in manifest.entries
        if (entry.logical_key, entry.state_kind, entry.tp_lane) not in states
    ]
    if missing:
        raise RuntimeError(f"manifest entries are not bound to active tensors: {missing[:3]}")
    return parameters, states


def _combined_manifest(
    model_manifest: StateManifest,
    optimizer_schema: OptimizerStateSchema | None,
) -> StateManifest:
    entries = list(model_manifest.entries)
    if optimizer_schema is not None:
        entries.extend(optimizer_schema.tensor_entries)
    return StateManifest(model_manifest.rank, tuple(entries), model_manifest.committed_step)


def capture_context_state(context: Any) -> RecoverySnapshot:
    """Clone the last committed rank-local state before retiring a generation."""

    model_manifest = version_manifest(context.partition.manifest, context.committed_step)
    named_parameters, model_tensors = logical_state_bindings(context.model, model_manifest)
    tensors = {key: value.detach().contiguous().clone() for key, value in model_tensors.items()}
    parameter_entries = {
        entry.logical_key: entry
        for entry in model_manifest.entries
        if entry.state_kind == "parameter"
    }
    optimizer_schema = None
    if context.optimizer is not None:
        optimizer_schema, optimizer_tensors = serialize_optimizer_state(
            context.optimizer,
            named_parameters,
            owner_rank=model_manifest.rank,
            committed_step=context.committed_step,
            parameter_entries=parameter_entries,
        )
        lanes = {entry.logical_key: entry.tp_lane for entry in optimizer_schema.tensor_entries}
        tensors.update(
            {
                (key, "optimizer", lanes[key]): value.detach().contiguous().clone()
                for key, value in optimizer_tensors.items()
            }
        )
    manifest = _combined_manifest(model_manifest, optimizer_schema)
    scheduler_state = (
        copy.deepcopy(context.scheduler.state_dict())
        if context.scheduler is not None and hasattr(context.scheduler, "state_dict")
        else None
    )
    scaler_state = (
        copy.deepcopy(context.scaler.state_dict())
        if context.scaler is not None and hasattr(context.scaler, "state_dict")
        else None
    )
    return RecoverySnapshot(
        manifest,
        tensors,
        optimizer_schema,
        scheduler_state,
        scaler_state,
        context.committed_step,
    )


def _distributed_identity(fallback_rank: int) -> tuple[Any, int, int]:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist, dist.get_rank(), dist.get_world_size()
    except (ImportError, RuntimeError):
        pass
    return None, fallback_rank, 1


def _reown_manifest(manifest: StateManifest, rank: int) -> StateManifest:
    return StateManifest(
        rank,
        tuple(replace(entry, owner_rank=rank) for entry in manifest.entries),
        manifest.committed_step,
    )


def _reown_schema(schema: OptimizerStateSchema | None, rank: int) -> OptimizerStateSchema | None:
    if schema is None:
        return None
    return replace(
        schema,
        tensor_entries=tuple(replace(entry, owner_rank=rank) for entry in schema.tensor_entries),
    )


def _gather_metadata(
    snapshot: RecoverySnapshot | None,
    *,
    rank: int,
    dist: Any,
    world_size: int,
) -> tuple[RecoveryMetadata, ...]:
    local = (
        None
        if snapshot is None
        else RecoveryMetadata(
            _reown_manifest(snapshot.manifest, rank),
            _reown_schema(snapshot.optimizer_schema, rank),
            snapshot.scheduler_state,
            snapshot.scaler_state,
            snapshot.committed_step,
        )
    )
    if dist is None:
        return () if local is None else (local,)
    gathered: list[RecoveryMetadata | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return tuple(item for item in gathered if item is not None)


def _dtype(value: str) -> torch.dtype:
    name = value.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported state dtype {value!r}")
    return dtype


def _copy_retained(
    old_manifest: StateManifest,
    new_manifest: StateManifest,
    sources: Mapping[StateTensorKey, torch.Tensor],
    destinations: MutableMapping[StateTensorKey, torch.Tensor],
) -> set[StateTensorKey]:
    old_entries = {
        (entry.logical_key, entry.state_kind, entry.tp_lane): entry
        for entry in old_manifest.entries
    }
    copied = set()
    for entry in new_manifest.entries:
        key = (entry.logical_key, entry.state_kind, entry.tp_lane)
        source_entry = old_entries.get(key)
        if source_entry is None or source_entry.shard_identity != entry.shard_identity:
            continue
        destinations[key].copy_(sources[key])
        copied.add(key)
    return copied


def restore_context_state(context: Any, snapshot: RecoverySnapshot) -> RecoveryReport:
    """Redistribute a committed snapshot into the context's active generation."""

    dist, rank, world_size = _distributed_identity(context.owner_plan.rank)
    if context.compiled.world_size != world_size:
        raise RuntimeError(
            f"activated plan expects world_size={context.compiled.world_size}, "
            f"but the new WORLD has {world_size} ranks"
        )
    metadata = _gather_metadata(snapshot, rank=rank, dist=dist, world_size=world_size)
    steps = {item.committed_step for item in metadata}
    if steps != {snapshot.committed_step}:
        raise RuntimeError(f"surviving ranks disagree on committed step: {sorted(steps)}")

    model_manifest = version_manifest(context.partition.manifest, snapshot.committed_step)
    if model_manifest.rank != rank:
        model_manifest = _reown_manifest(model_manifest, rank)
    named_parameters, model_destinations = logical_state_bindings(context.model, model_manifest)
    parameter_entries = {
        entry.logical_key: entry
        for entry in model_manifest.entries
        if entry.state_kind == "parameter"
    }

    optimizer_schema = None
    optimizer_destinations: dict[StateTensorKey, torch.Tensor] = {}
    schemas = tuple(item.optimizer_schema for item in metadata if item.optimizer_schema is not None)
    if snapshot.optimizer_schema is not None:
        if not schemas or context._optimizer_factory is None:
            raise RuntimeError("optimizer recovery metadata/factory is unavailable")
        optimizer_schema = optimizer_schema_for_partition(
            schemas,
            parameter_entries,
            owner_rank=rank,
            committed_step=snapshot.committed_step,
        )
        for entry in optimizer_schema.tensor_entries:
            optimizer_destinations[(entry.logical_key, "optimizer", entry.tp_lane)] = torch.empty(
                entry.local_shape, dtype=_dtype(entry.dtype), device=context.device
            )

    new_manifest = _combined_manifest(model_manifest, optimizer_schema)
    old_manifests = tuple(item.manifest for item in metadata)
    schedule = plan_state_redistribution(
        old_manifests,
        (new_manifest,),
        chunk_bytes=context.config.state_transfer_chunk_bytes,
        round_bytes=getattr(
            context.config,
            "state_transfer_round_bytes",
            context.config.state_transfer_chunk_bytes * world_size,
        ),
        alignment=context.config.transfer_alignment_bytes,
    )
    if dist is not None:
        hashes: list[str | None] = [None] * world_size
        dist.all_gather_object(hashes, schedule.schedule_hash)
        if set(hashes) != {schedule.schedule_hash}:
            raise RuntimeError("ranks derived different state-transfer schedules")

    destinations: dict[StateTensorKey, torch.Tensor] = dict(model_destinations)
    destinations.update(optimizer_destinations)
    local_old_manifest = metadata[rank].manifest
    copied = _copy_retained(local_old_manifest, new_manifest, snapshot.tensors, destinations)
    transfer_metrics = execute_transfer_schedule(
        schedule,
        rank=rank,
        world_size=world_size,
        sources=snapshot.tensors,
        destinations=destinations,
        device=context.device,
    )
    incoming = {
        (item.logical_key, item.state_kind, item.tp_lane)
        for item in schedule.transfers
        if item.destination_rank == rank
    }
    missing = set(destinations) - copied - incoming
    if missing:
        raise RuntimeError(f"recovery left state uninitialized: {sorted(missing)[:3]}")

    if optimizer_schema is not None:
        context.optimizer = context._optimizer_factory(context.model.parameters())
        restore_optimizer_state(
            context.optimizer,
            optimizer_schema,
            {
                entry.logical_key: optimizer_destinations[
                    (entry.logical_key, "optimizer", entry.tp_lane)
                ]
                for entry in optimizer_schema.tensor_entries
            },
            named_parameters,
        )
        context.scheduler = (
            context._scheduler_factory(context.optimizer)
            if context._scheduler_factory is not None
            else None
        )
        canonical = metadata[0]
        if context.scheduler is not None and canonical.scheduler_state is not None:
            context.scheduler.load_state_dict(copy.deepcopy(canonical.scheduler_state))
        if context.scaler is not None and canonical.scaler_state is not None:
            context.scaler.load_state_dict(copy.deepcopy(canonical.scaler_state))
    if dist is not None:
        dist.barrier()
    return RecoveryReport(
        schedule=schedule,
        source_bytes=schedule.per_source_bytes,
        destination_bytes=schedule.per_destination_bytes,
        link_class_bytes=schedule.per_link_class_bytes,
        actual_source_bytes=transfer_metrics.actual_source_bytes,
        actual_destination_bytes=transfer_metrics.actual_destination_bytes,
        actual_link_class_bytes=transfer_metrics.actual_link_class_bytes,
        round_durations=transfer_metrics.round_durations,
    )


__all__ = [
    "RecoveryReport",
    "RecoverySnapshot",
    "capture_context_state",
    "logical_state_bindings",
    "restore_context_state",
    "version_manifest",
]
