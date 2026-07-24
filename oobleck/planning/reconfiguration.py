"""Pure simple/borrow/merge membership reconfiguration planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from oobleck.planning.composer import (
    allocate_microbatches,
    compose_templates,
    template_for_resources,
)
from oobleck.types import PipelineInstance, PipelineTemplate, RecoveryUnavailable


@dataclass(frozen=True, slots=True)
class ReconfigurationResult:
    """A complete survivor assignment plus recovery-strategy accounting."""

    instances: tuple[PipelineInstance, ...]
    strategies: tuple[str, ...]
    retained_nodes: int
    moved_nodes: int
    retained_state_bytes: int


def _instance(
    identity: str, nodes: Iterable[str], templates: Sequence[PipelineTemplate]
) -> PipelineInstance | None:
    """Create an ordered instance when its resource count is supported."""

    selected_nodes = tuple(sorted(nodes))
    template = template_for_resources(templates, len(selected_nodes))
    return None if template is None else PipelineInstance(identity, template, selected_nodes)


def reconfigure_pipelines(
    previous: Sequence[PipelineInstance],
    surviving_node_ids: Iterable[str],
    templates: Sequence[PipelineTemplate],
    total_microbatches: int,
    fault_tolerance_threshold: int,
    *,
    state_bytes_by_node: Mapping[str, int] | None = None,
) -> ReconfigurationResult:
    """Apply simple, borrow, and merge with throughput/state-stable ties."""

    survivors = set(surviving_node_ids)
    weights = dict(state_bytes_by_node or {})
    if any(value < 0 for value in weights.values()):
        raise ValueError("state byte estimates must be non-negative")
    known = {node for item in previous for node in item.node_ids}
    old_owner = {node: item.instance_id for item in previous for node in item.node_ids}

    def retained_bytes(identity: str, nodes: Iterable[str]) -> int:
        """Score state that stays under the same logical pipeline identity."""

        return sum(weights.get(node, 1) for node in nodes if old_owner.get(node) == identity)

    if not survivors <= known:
        instances = compose_templates(
            templates,
            sorted(survivors),
            total_microbatches,
            fault_tolerance_threshold,
            previous_instances=previous,
            state_bytes_by_node=weights,
        )
        retained = sum(
            1
            for item in instances
            for node in item.node_ids
            if old_owner.get(node) == item.instance_id
        )
        retained_state = sum(retained_bytes(item.instance_id, item.node_ids) for item in instances)
        known_survivors = survivors.intersection(known)
        return ReconfigurationResult(
            instances,
            ("join",),
            retained,
            len(known_survivors) - retained,
            retained_state,
        )

    groups: list[list] = [
        [item.instance_id, set(item.node_ids) & survivors]
        for item in sorted(previous, key=lambda value: value.instance_id)
        if set(item.node_ids) & survivors
    ]
    if not groups:
        raise RecoveryUnavailable("no nodes survived the membership change")

    strategies: list[str] = []
    old_by_identity = {item.instance_id: item for item in previous}
    if any(len(nodes) != len(old_by_identity[identity].node_ids) for identity, nodes in groups):
        strategies.append("simple")
    minimum = min(item.resource_count for item in templates)

    while True:
        undersized = next(
            (
                group
                for group in groups
                if template_for_resources(templates, len(group[1])) is None
                and len(group[1]) < minimum
            ),
            None,
        )
        if undersized is None:
            break
        donors = [
            group
            for group in groups
            if group is not undersized
            and template_for_resources(templates, len(group[1]) - 1) is not None
        ]
        if not donors:
            break
        donor = min(
            donors,
            key=lambda group: (
                -len(group[1]),
                min((weights.get(node, 1), node) for node in group[1]),
                group[0],
            ),
        )
        moved = min(donor[1], key=lambda node: (weights.get(node, 1), node))
        donor[1].remove(moved)
        undersized[1].add(moved)
        if "borrow" not in strategies:
            strategies.append("borrow")

    while any(template_for_resources(templates, len(group[1])) is None for group in groups):
        target = min(
            (group for group in groups if template_for_resources(templates, len(group[1])) is None),
            key=lambda group: (len(group[1]), group[0]),
        )
        partners = [group for group in groups if group is not target]
        if not partners:
            raise RecoveryUnavailable(
                f"surviving group of {len(target[1])} nodes has no supported template"
            )

        def partner_objective(partner: list) -> tuple[object, ...]:
            """Rank merge partners by throughput, retained state, size, and ID."""

            combined = target[1] | partner[1]
            template = template_for_resources(templates, len(combined))
            predicted = float("inf") if template is None else template.iteration_time(1)
            best_retained = max(
                retained_bytes(target[0], combined),
                retained_bytes(partner[0], combined),
            )
            return predicted, -best_retained, len(partner[1]), partner[0]

        partner = min(partners, key=partner_objective)
        combined = target[1] | partner[1]
        target_score = retained_bytes(target[0], combined)
        partner_score = retained_bytes(partner[0], combined)
        if partner_score > target_score or (
            partner_score == target_score and partner[0] < target[0]
        ):
            target[0] = partner[0]
        target[1].update(partner[1])
        groups.remove(partner)
        if "merge" not in strategies:
            strategies.append("merge")

    instances = []
    for identity, nodes in sorted(groups):
        item = _instance(identity, nodes, templates)
        if item is None:
            raise RecoveryUnavailable(
                f"pipeline {identity} has unsupported resource count {len(nodes)}"
            )
        instances.append(item)

    required = fault_tolerance_threshold + 1
    if len(instances) < required:
        raise RecoveryUnavailable(
            f"recovery produced {len(instances)} replicas but {required} are required"
        )
    allocated = allocate_microbatches(instances, total_microbatches)
    assigned = [node for item in allocated for node in item.node_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != survivors:
        raise AssertionError("reconfiguration did not assign each survivor exactly once")
    retained = sum(
        1 for item in allocated for node in item.node_ids if old_owner.get(node) == item.instance_id
    )
    retained_state = sum(retained_bytes(item.instance_id, item.node_ids) for item in allocated)
    return ReconfigurationResult(
        allocated,
        tuple(strategies or ["unchanged"]),
        retained,
        len(survivors) - retained,
        retained_state,
    )
