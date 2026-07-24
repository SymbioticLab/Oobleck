"""Logical-layer/TP-lane gradient synchronization topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from oobleck.recovery import logical_state_bindings
from oobleck.state import StateManifest
from oobleck.types import OobleckExecutionPlan


@dataclass(frozen=True, slots=True)
class GradientSyncGroup:
    """Replica ranks and sample weights for one logical parameter shard."""

    logical_key: str
    tp_lane: int
    placements: tuple[str, ...]
    ranks: tuple[int, ...]
    sample_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require deterministic membership and a normalized weighted reduction."""

        if len(self.ranks) != len(self.sample_weights) or not self.ranks:
            raise ValueError("gradient group ranks and sample weights must align")
        if tuple(sorted(self.ranks)) != self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("gradient group ranks must be sorted and unique")
        if abs(sum(self.sample_weights) - 1.0) > 1e-9:
            raise ValueError("gradient sample weights must sum to one")


def build_gradient_sync_groups(
    manifests: Sequence[StateManifest],
    rank_to_pipeline: Mapping[int, str],
    pipeline_sample_counts: Mapping[str, int],
) -> tuple[GradientSyncGroup, ...]:
    """Group parameter owners by logical identity, TP lane, and placement.

    Each pipeline contributes in proportion to the number of samples it processed,
    which preserves global-batch gradient semantics under heterogeneous allocation.
    """

    owners: dict[tuple[str, int, tuple[str, ...]], set[int]] = {}
    for manifest in manifests:
        for entry in manifest.entries:
            if entry.state_kind != "parameter":
                continue
            logical_key = entry.shared_state_id or entry.logical_key
            owners.setdefault((logical_key, entry.tp_lane, entry.placements), set()).add(
                manifest.rank
            )
    result = []
    for (key, lane, placements), ranks_set in sorted(owners.items()):
        ranks = tuple(sorted(ranks_set))
        counts = [pipeline_sample_counts[rank_to_pipeline[rank]] for rank in ranks]
        total = sum(counts)
        if total <= 0:
            raise ValueError(f"logical parameter {key} has no contributing samples")
        result.append(
            GradientSyncGroup(
                key,
                lane,
                placements,
                ranks,
                tuple(count / total for count in counts),
            )
        )
    return tuple(result)


def create_process_groups(groups: Sequence[GradientSyncGroup]):
    """Collectively create groups in deterministic logical-key order."""

    import torch.distributed as dist

    ordered = sorted(
        groups,
        key=lambda item: (item.logical_key, item.tp_lane, item.placements, item.ranks),
    )
    return {group: dist.new_group(ranks=list(group.ranks)) for group in ordered}


def parameter_bindings_by_sync_key(
    named_parameters: Mapping[str, torch.nn.Parameter],
    manifest: StateManifest,
) -> dict[str, torch.nn.Parameter]:
    """Alias tied/shared state identities to their local parameter object."""

    bindings = dict(named_parameters)
    for entry in manifest.entries:
        if entry.state_kind != "parameter" or entry.shared_state_id is None:
            continue
        parameter = named_parameters.get(entry.logical_key)
        if parameter is None:
            continue
        existing = bindings.setdefault(entry.shared_state_id, parameter)
        if existing is not parameter:
            raise RuntimeError(
                f"shared state {entry.shared_state_id!r} maps to distinct local tensors"
            )
    return bindings


class HeterogeneousGradientSynchronizer:
    """Sample-weighted logical-parameter reduction across pipeline replicas."""

    def __init__(self, bindings, process_groups) -> None:
        """Capture local parameters and the deterministically created groups."""

        self._bindings = tuple(bindings)
        self._process_groups = process_groups
        self._closed = False

    def sync(self) -> None:
        """Sample-weight and all-reduce every present logical gradient.

        A presence collective distinguishes an unused parameter from a missing
        local gradient; Gloo reductions stage CUDA payloads through CPU storage.
        """

        if self._closed:
            raise RuntimeError("gradient synchronizer is closed")
        import torch.distributed as dist

        for group, process_group, parameter, sample_weight in self._bindings:
            if len(group.ranks) == 1:
                continue
            gradient = parameter.grad
            has_gradient = gradient is not None
            local_parameter = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            local_gradient = (
                gradient.to_local()
                if gradient is not None and hasattr(gradient, "to_local")
                else gradient
            )
            backend = dist.get_backend(process_group)
            collective_device = (
                torch.device("cpu")
                if backend == "gloo" and local_parameter.device.type == "cuda"
                else local_parameter.device
            )
            present = torch.tensor(int(has_gradient), dtype=torch.int32, device=collective_device)
            dist.all_reduce(present, group=process_group)
            if int(present.item()) == 0:
                continue
            payload = (
                torch.zeros_like(local_parameter, device=collective_device)
                if local_gradient is None
                else local_gradient.detach().to(collective_device).mul(sample_weight)
            )
            dist.all_reduce(payload, group=process_group)
            if gradient is None:
                parameter.grad = torch.zeros_like(parameter)
                gradient = parameter.grad
                local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
            assert local_gradient is not None
            local_gradient.copy_(payload.to(local_gradient.device))

    def close(self) -> None:
        """Drop generation-local parameter and process-group references."""

        self._bindings = ()
        self._process_groups = {}
        self._closed = True


def activate_gradient_synchronizer(
    model: torch.nn.Module,
    local_manifest: StateManifest,
    execution_plan: OobleckExecutionPlan,
    *,
    microbatch_size: int,
) -> HeterogeneousGradientSynchronizer | None:
    """Gather ownership, create every logical group, and bind local parameters."""

    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    world_size = dist.get_world_size()
    manifests: list[StateManifest | None] = [None] * world_size
    dist.all_gather_object(manifests, local_manifest)
    if any(item is None for item in manifests):
        raise RuntimeError("gradient ownership manifest exchange was incomplete")
    rank_to_pipeline = {
        rank: instance.instance_id
        for instance in execution_plan.instances
        for stage_ranks in instance.ranks
        for rank in stage_ranks
    }
    sample_counts = {
        instance.instance_id: instance.microbatches * microbatch_size
        for instance in execution_plan.instances
    }
    groups = build_gradient_sync_groups(
        tuple(item for item in manifests if item is not None),
        rank_to_pipeline,
        sample_counts,
    )
    process_groups = create_process_groups(groups)
    named_parameters, _ = logical_state_bindings(model, local_manifest)
    sync_parameters = parameter_bindings_by_sync_key(named_parameters, local_manifest)
    rank = dist.get_rank()
    bindings = []
    for group in groups:
        if rank not in group.ranks:
            continue
        parameter = sync_parameters.get(group.logical_key)
        if parameter is None:
            raise RuntimeError(
                f"rank {rank} owns {group.logical_key!r} in topology but not in its model"
            )
        bindings.append(
            (
                group,
                process_groups[group],
                parameter,
                group.sample_weights[group.ranks.index(rank)],
            )
        )
    return HeterogeneousGradientSynchronizer(bindings, process_groups)


__all__ = [
    "GradientSyncGroup",
    "HeterogeneousGradientSynchronizer",
    "activate_gradient_synchronizer",
    "build_gradient_sync_groups",
    "create_process_groups",
    "parameter_bindings_by_sync_key",
]
