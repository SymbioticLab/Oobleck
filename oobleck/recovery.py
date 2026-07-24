"""Public recovery coordinator with all-rank destination planning."""

from __future__ import annotations

from typing import Any

from oobleck.recovery_base import *  # noqa: F403
from oobleck.recovery_base import __all__ as _base_all
from oobleck.recovery_base import (
    RecoveryReport,
    RecoverySnapshot,
    _combined_manifest,
    _copy_retained,
    _distributed_identity,
    _dtype,
    _gather_metadata,
    _reown_manifest,
    logical_state_bindings,
    version_manifest,
)
from oobleck.optimization import optimizer_schema_for_partition, restore_optimizer_state
from oobleck.state import StateManifest, plan_state_redistribution
from oobleck.state_transfer import execute_transfer_schedule

import copy
import time
import torch


def _gather_new_manifests(
    manifest: StateManifest, *, dist: Any, world_size: int
) -> tuple[StateManifest, ...]:
    """Exchange every destination manifest before deriving a global schedule."""

    if dist is None:
        return (manifest,)
    gathered: list[StateManifest | None] = [None] * world_size
    dist.all_gather_object(gathered, manifest)
    if any(item is None for item in gathered):
        raise RuntimeError("destination manifest exchange was incomplete")
    return tuple(item for item in gathered if item is not None)


def _rank_to_node(context: Any) -> dict[int, str] | None:
    """Project the active execution rank map into transfer-locality metadata."""

    execution_plan = getattr(context, "execution_plan", None)
    rank_map = getattr(execution_plan, "rank_map", None)
    if rank_map is None:
        return None
    return {rank: node_id for node_id, ranks in rank_map for rank in ranks}


def restore_context_state(context: Any, snapshot: RecoverySnapshot | None) -> RecoveryReport:
    """Restore one agreed committed transaction into the entire replacement generation.

    Surviving ranks exchange tensor-free source metadata, agree on one committed step, and build
    model/optimizer destination manifests for every new rank—including added workers with no
    snapshot. All ranks derive and checksum one global redistribution schedule before retained
    local shards are copied and remote chunks execute collectively. Completeness is verified for
    every destination, then optimizer groups/slots, scheduler, scaler, and committed step are
    reconstructed. A final WORLD barrier prevents any rank from training on partially restored
    state; phase and load metrics describe the completed recovery.
    """

    started = time.perf_counter()
    dist, rank, world_size = _distributed_identity(context.owner_plan.rank)
    if context.compiled.world_size != world_size:
        raise RuntimeError(
            f"activated plan expects world_size={context.compiled.world_size}, "
            f"but the new WORLD has {world_size} ranks"
        )
    metadata = _gather_metadata(snapshot, rank=rank, dist=dist, world_size=world_size)
    if not metadata:
        raise RuntimeError("no surviving committed state source participated in recovery")
    metadata_finished = time.perf_counter()
    steps = {item.committed_step for item in metadata}
    if len(steps) != 1:
        raise RuntimeError(f"surviving ranks disagree on committed step: {sorted(steps)}")
    committed_step = next(iter(steps))

    model_manifest = version_manifest(context.partition.manifest, committed_step)
    if model_manifest.rank != rank:
        model_manifest = _reown_manifest(model_manifest, rank)
    named_parameters, model_destinations = logical_state_bindings(context.model, model_manifest)
    parameter_entries = {
        entry.logical_key: entry
        for entry in model_manifest.entries
        if entry.state_kind == "parameter"
    }

    optimizer_schema = None
    optimizer_destinations = {}
    schemas = tuple(item.optimizer_schema for item in metadata if item.optimizer_schema is not None)
    if schemas:
        if not schemas or context._optimizer_factory is None:
            raise RuntimeError("optimizer recovery metadata/factory is unavailable")
        optimizer_schema = optimizer_schema_for_partition(
            schemas,
            parameter_entries,
            owner_rank=rank,
            committed_step=committed_step,
        )
        for entry in optimizer_schema.tensor_entries:
            optimizer_destinations[(entry.logical_key, "optimizer", entry.tp_lane)] = torch.empty(
                entry.local_shape, dtype=_dtype(entry.dtype), device=context.device
            )

    new_manifest = _combined_manifest(model_manifest, optimizer_schema)
    new_manifests = _gather_new_manifests(new_manifest, dist=dist, world_size=world_size)
    schedule = plan_state_redistribution(
        tuple(item.manifest for item in metadata),
        new_manifests,
        chunk_bytes=context.config.state_transfer_chunk_bytes,
        round_bytes=getattr(
            context.config,
            "state_transfer_round_bytes",
            context.config.state_transfer_chunk_bytes * world_size,
        ),
        alignment=context.config.transfer_alignment_bytes,
        rank_to_node=_rank_to_node(context),
    )
    if dist is not None:
        hashes: list[str | None] = [None] * world_size
        dist.all_gather_object(hashes, schedule.schedule_hash)
        if set(hashes) != {schedule.schedule_hash}:
            raise RuntimeError("ranks derived different state-transfer schedules")
    planning_finished = time.perf_counter()

    destinations = dict(model_destinations)
    destinations.update(optimizer_destinations)
    source_tensors = snapshot.tensors if snapshot is not None else {}
    local_old_manifest = next(
        (item.manifest for item in metadata if item.manifest.rank == rank), None
    )
    with torch.no_grad():
        copied = (
            _copy_retained(local_old_manifest, new_manifest, source_tensors, destinations)
            if local_old_manifest is not None
            else set()
        )
        transfer_metrics = execute_transfer_schedule(
            schedule,
            rank=rank,
            world_size=world_size,
            sources=source_tensors,
            destinations=destinations,
            device=context.device,
        )
    transfer_finished = time.perf_counter()
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
    sampler_states = metadata[0].sampler_states
    if any(item.sampler_states != sampler_states for item in metadata[1:]):
        raise RuntimeError("surviving ranks disagree on committed sampler state")
    if sampler_states:
        loaders = getattr(context, "_loaders", ())
        if len(sampler_states) != len(loaders):
            raise RuntimeError("recovery sampler metadata does not match configured loaders")
        for loader, state in zip(loaders, sampler_states):
            loader.sampler.load_state_dict(copy.deepcopy(state))
    context.committed_step = committed_step
    if dist is not None:
        dist.barrier()
    finished = time.perf_counter()
    return RecoveryReport(
        schedule=schedule,
        source_bytes=schedule.per_source_bytes,
        destination_bytes=schedule.per_destination_bytes,
        link_class_bytes=schedule.per_link_class_bytes,
        actual_source_bytes=transfer_metrics.actual_source_bytes,
        actual_destination_bytes=transfer_metrics.actual_destination_bytes,
        actual_link_class_bytes=transfer_metrics.actual_link_class_bytes,
        round_durations=transfer_metrics.round_durations,
        metadata_exchange_seconds=metadata_finished - started,
        planning_seconds=planning_finished - metadata_finished,
        transfer_seconds=transfer_finished - planning_finished,
        restoration_seconds=finished - transfer_finished,
        total_seconds=finished - started,
    )


__all__ = [*_base_all, "restore_context_state"]
