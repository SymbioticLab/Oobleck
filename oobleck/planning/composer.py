"""Deterministic heterogeneous template composition and batch allocation."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from oobleck.types import PipelineInstance, PipelineTemplate, RecoveryUnavailable


def allocate_microbatches(
    instances: Sequence[PipelineInstance], total_microbatches: int
) -> tuple[PipelineInstance, ...]:
    """Allocate integer microbatches to minimize predicted iteration makespan.

    The search first finds the smallest feasible iteration-time threshold under
    each template's memory capacity. Every replica receives one microbatch, then
    remaining work is assigned greedily by its next completion time with stable
    instance-ID tie breaks. The returned allocation always consumes the complete
    fixed global batch without silently disabling a pipeline.
    """

    if not instances:
        raise RecoveryUnavailable("cannot allocate a batch without a pipeline")
    if total_microbatches < len(instances):
        raise RecoveryUnavailable(
            f"{total_microbatches} microbatches cannot keep {len(instances)} "
            "pipeline replicas active"
        )

    candidates: set[float] = set()
    for instance in instances:
        limit = min(
            total_microbatches,
            instance.template.max_microbatches or total_microbatches,
        )
        candidates.update(instance.template.iteration_time(count) for count in range(1, limit + 1))
    candidates.discard(math.inf)

    threshold: float | None = None
    capacities: list[int] = []
    for value in sorted(candidates):
        trial = []
        for instance in instances:
            cap = 0
            max_count = min(
                total_microbatches,
                instance.template.max_microbatches or total_microbatches,
            )
            for count in range(1, max_count + 1):
                if instance.template.iteration_time(count) <= value:
                    cap = count
            trial.append(cap)
        if all(trial) and sum(trial) >= total_microbatches:
            threshold, capacities = value, trial
            break
    if threshold is None:
        raise RecoveryUnavailable("global batch exceeds template microbatch capacity")

    allocation = [1] * len(instances)
    remaining = total_microbatches - len(instances)
    stable_order = sorted(range(len(instances)), key=lambda index: instances[index].instance_id)
    while remaining:
        choices = [index for index in stable_order if allocation[index] < capacities[index]]
        if not choices:
            raise RecoveryUnavailable("internal batch allocation capacity mismatch")
        chosen = min(
            choices,
            key=lambda index: (
                instances[index].template.iteration_time(allocation[index] + 1),
                instances[index].instance_id,
            ),
        )
        allocation[chosen] += 1
        remaining -= 1

    return tuple(
        replace(instance, microbatches=allocation[index])
        for index, instance in enumerate(instances)
    )


def _resource_compositions(
    templates: Sequence[PipelineTemplate], resources: int, minimum_replicas: int
) -> Iterable[tuple[PipelineTemplate, ...]]:
    """Enumerate stable template multisets that consume every available node."""

    ordered = sorted(templates, key=lambda item: (item.resource_count, item.template_id))

    def visit(
        start: int, remaining: int, selected: tuple[PipelineTemplate, ...]
    ) -> Iterable[tuple[PipelineTemplate, ...]]:
        """Backtrack through nondecreasing templates to avoid duplicates."""

        if remaining == 0:
            if len(selected) >= minimum_replicas:
                yield selected
            return
        for index in range(start, len(ordered)):
            template = ordered[index]
            if template.resource_count <= remaining:
                yield from visit(index, remaining - template.resource_count, (*selected, template))

    return visit(0, resources, ())


def _assign_composition(
    composition: Sequence[PipelineTemplate],
    nodes: tuple[str, ...],
    previous: Sequence[PipelineInstance],
    state_bytes: Mapping[str, int],
) -> tuple[tuple[PipelineInstance, ...], int, int]:
    """Map a template composition to nodes while preserving maximum local state.

    A dynamic-programming match pairs new composition slots with surviving old
    pipeline identities using retained-state bytes as the primary score. Reserved
    survivors stay with the winning identity; remaining nodes fill deficits in
    stable order. The result also reports retained bytes and survivor movement so
    higher-level composition can apply deterministic recovery tie breaks.
    """

    surviving = set(nodes)
    old = tuple(
        item
        for item in sorted(previous, key=lambda value: value.instance_id)
        if surviving.intersection(item.node_ids)
    )
    if not old:
        instances = []
        cursor = 0
        for index, template in enumerate(composition):
            selected = nodes[cursor : cursor + template.resource_count]
            cursor += template.resource_count
            instances.append(PipelineInstance(f"pipeline-{index:04d}", template, selected))
        return tuple(instances), 0, 0

    retained_by_pair: dict[tuple[int, int], int] = {}
    selected_by_pair: dict[tuple[int, int], tuple[str, ...]] = {}
    for slot, template in enumerate(composition):
        for old_index, instance in enumerate(old):
            candidates = sorted(
                surviving.intersection(instance.node_ids),
                key=lambda node: (-state_bytes.get(node, 1), node),
            )[: template.resource_count]
            selected_by_pair[slot, old_index] = tuple(sorted(candidates))
            retained_by_pair[slot, old_index] = sum(state_bytes.get(node, 1) for node in candidates)

    @lru_cache(maxsize=None)
    def choose(slot: int, used: int) -> tuple[int, tuple[int, ...]]:
        """Match old identities to new slots while maximizing retained state."""

        if slot == len(composition):
            return 0, ()
        options: list[tuple[int, tuple[int, ...]]] = []
        score, suffix = choose(slot + 1, used)
        options.append((score, (-1, *suffix)))
        for old_index in range(len(old)):
            if used & (1 << old_index):
                continue
            suffix_score, suffix = choose(slot + 1, used | (1 << old_index))
            options.append(
                (
                    retained_by_pair[slot, old_index] + suffix_score,
                    (old_index, *suffix),
                )
            )
        best_score = max(item[0] for item in options)
        return min((item for item in options if item[0] == best_score), key=lambda item: item[1])

    retained, assignment = choose(0, 0)
    reserved = {
        node
        for slot, old_index in enumerate(assignment)
        if old_index >= 0
        for node in selected_by_pair[slot, old_index]
    }
    remaining = [node for node in nodes if node not in reserved]
    old_ids = {item.instance_id for item in old}
    instances = []
    used_ids: set[str] = set()
    for slot, (template, old_index) in enumerate(zip(composition, assignment)):
        if old_index >= 0:
            identity = old[old_index].instance_id
            selected = list(selected_by_pair[slot, old_index])
        else:
            identity = f"pipeline-{slot:04d}"
            suffix = 0
            while identity in old_ids or identity in used_ids:
                suffix += 1
                identity = f"pipeline-new-{slot:04d}-{suffix:02d}"
            selected = []
        deficit = template.resource_count - len(selected)
        selected.extend(remaining[:deficit])
        del remaining[:deficit]
        used_ids.add(identity)
        instances.append(PipelineInstance(identity, template, tuple(sorted(selected))))
    if remaining:
        raise AssertionError("composition assignment left nodes unassigned")
    known_survivors = surviving.intersection(node for item in previous for node in item.node_ids)
    moved = sum(
        1
        for item in instances
        for node in item.node_ids
        if node in known_survivors
        and not any(
            old_item.instance_id == item.instance_id and node in old_item.node_ids
            for old_item in old
        )
    )
    return tuple(instances), retained, moved


def compose_templates(
    templates: Sequence[PipelineTemplate],
    node_ids: Sequence[str],
    total_microbatches: int,
    fault_tolerance_threshold: int,
    *,
    previous_instances: Sequence[PipelineInstance] = (),
    state_bytes_by_node: Mapping[str, int] | None = None,
) -> tuple[PipelineInstance, ...]:
    """Choose a complete heterogeneous composition with deterministic objectives.

    Every multiset of templates that exactly consumes membership and satisfies the
    replica threshold is considered. Feasible candidates receive an integer batch
    allocation, then compare by predicted makespan, retained state, moved nodes,
    template IDs, and concrete ownership. This ordering lets all workers derive an
    identical plan while preferring throughput before recovery convenience.
    """

    if not templates:
        raise RecoveryUnavailable("no pipeline templates are available")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node IDs must be unique")
    if fault_tolerance_threshold < 0:
        raise ValueError("fault_tolerance_threshold must be non-negative")
    weights = dict(state_bytes_by_node or {})
    if any(value < 0 for value in weights.values()):
        raise ValueError("state byte estimates must be non-negative")
    required_replicas = fault_tolerance_threshold + 1
    best: tuple[tuple[object, ...], tuple[PipelineInstance, ...]] | None = None
    nodes = tuple(sorted(node_ids))

    for composition in _resource_compositions(templates, len(nodes), required_replicas):
        instances, retained, moved = _assign_composition(
            composition, nodes, previous_instances, weights
        )
        try:
            allocated = allocate_microbatches(instances, total_microbatches)
        except RecoveryUnavailable:
            continue
        objective: tuple[object, ...] = (
            max(instance.template.iteration_time(instance.microbatches) for instance in allocated),
            -retained,
            moved,
            tuple(instance.template.template_id for instance in allocated),
            tuple((item.instance_id, item.node_ids) for item in allocated),
        )
        if best is None or objective < best[0]:
            best = objective, allocated

    if best is None:
        raise RecoveryUnavailable(
            f"{len(nodes)} nodes cannot form {required_replicas} supported "
            f"pipelines for {total_microbatches} microbatches"
        )
    return best[1]


def gradient_sample_weights(
    instances: Sequence[PipelineInstance], microbatch_size: int
) -> dict[str, float]:
    """Return per-pipeline weights whose sample contribution sums to one."""

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be >= 1")
    total = sum(instance.microbatches * microbatch_size for instance in instances)
    if total == 0:
        raise ValueError("at least one microbatch must be allocated")
    return {
        instance.instance_id: instance.microbatches * microbatch_size / total
        for instance in instances
    }


def template_for_resources(
    templates: Sequence[PipelineTemplate], resources: int
) -> PipelineTemplate | None:
    """Pick the fastest supported template with a stable tie break."""

    candidates = [item for item in templates if item.resource_count == resources]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.iteration_time(1), item.template_id))
