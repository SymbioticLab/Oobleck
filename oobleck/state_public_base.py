"""Public state API with homogeneous-cluster source striping."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from oobleck.state_base import (
    LogicalStateEntry,
    StateManifest,
    StateUnavailable,
    Transfer,
    TransferSchedule,
    all_to_all_single_compat,
)
from oobleck.state_base import plan_state_redistribution as _plan


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
    """Refine base transfer planning so homogeneous links stripe across source pairs.

    Without an operator classifier, each source/destination pair receives an independent temporary
    link bucket, with same-node pairs retaining their lower relative cost. This prevents aggregate
    ``network`` load from collapsing otherwise equal source choices. After planning, pair-specific
    labels are normalized back to public ``same-node``/``network`` classes and byte totals recomputed;
    explicit operator classifiers pass through unchanged.
    """

    normalize_default_links = link_classifier is None
    if link_classifier is None:

        def link_classifier(source: int, destination: int) -> tuple[str, float]:
            """Give each homogeneous pair an independent scheduler load bucket."""

            if rank_to_node and rank_to_node.get(source) == rank_to_node.get(destination):
                return f"same-node:{source}:{destination}", 0.25
            return f"network:{source}:{destination}", 1.0

    schedule = _plan(
        old_manifests,
        new_manifests,
        chunk_bytes=chunk_bytes,
        round_bytes=round_bytes,
        alignment=alignment,
        rank_to_node=rank_to_node,
        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        link_classifier=link_classifier,
    )
    if not normalize_default_links:
        return schedule
    transfers = tuple(
        replace(item, link_class=item.link_class.split(":", 1)[0]) for item in schedule.transfers
    )
    link_bytes: dict[str, int] = {}
    for item in transfers:
        link_bytes[item.link_class] = link_bytes.get(item.link_class, 0) + item.byte_count
    return TransferSchedule(
        transfers,
        schedule.retained,
        schedule.per_source_bytes,
        schedule.per_destination_bytes,
        tuple(sorted(link_bytes.items())),
    )


__all__ = [
    "LogicalStateEntry",
    "StateManifest",
    "StateUnavailable",
    "Transfer",
    "TransferSchedule",
    "all_to_all_single_compat",
    "plan_state_redistribution",
]
