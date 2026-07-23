"""Public stable-state and redistribution API."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from oobleck.state_public_base import (
    LogicalStateEntry,
    StateManifest,
    StateUnavailable,
    Transfer,
    TransferSchedule,
    all_to_all_single_compat,
)
from oobleck.state_public_base import plan_state_redistribution as _plan


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
    # In the topology-free model destination ingress is identical for every
    # candidate and must not erase the source-egress tie-break.  Infinite
    # destination bandwidth removes only that common term; measured bandwidth
    # supplied by an operator remains authoritative.
    bandwidth = bandwidth_bytes_per_s
    if bandwidth is None:
        bandwidth = {
            entry.owner_rank: 1.0 for manifest in old_manifests for entry in manifest.entries
        }
        bandwidth.update({manifest.rank: math.inf for manifest in new_manifests})
    return _plan(
        old_manifests,
        new_manifests,
        chunk_bytes=chunk_bytes,
        round_bytes=round_bytes,
        alignment=alignment,
        rank_to_node=rank_to_node,
        bandwidth_bytes_per_s=bandwidth,
        link_classifier=link_classifier,
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
