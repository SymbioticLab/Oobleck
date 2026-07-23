"""Transport-independent, generation-based membership state machine."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping


class StaleGeneration(RuntimeError):
    pass


class StaleSequence(RuntimeError):
    pass


class IncarnationMismatch(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    agent_id: str
    incarnation_id: str
    addresses: tuple[str, ...]
    gpu_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.agent_id or not self.incarnation_id:
            raise ValueError("agent_id and incarnation_id are required")
        if not self.addresses or not self.gpu_ids:
            raise ValueError("addresses and gpu_ids must be non-empty")


@dataclass(slots=True)
class _LiveNode:
    identity: NodeIdentity
    last_sequence: int
    lease_deadline: float


@dataclass(frozen=True, slots=True)
class MembershipSnapshot:
    generation: int
    nodes: tuple[NodeIdentity, ...]
    reasons: tuple[str, ...]
    snapshot_hash: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        payload = {
            "generation": self.generation,
            "nodes": [asdict(item) for item in self.nodes],
            "reasons": self.reasons,
        }
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("membership snapshot hash is invalid")
        object.__setattr__(self, "snapshot_hash", expected)


def membership_snapshot_from_payload(
    generation: int, payload: Mapping[str, object]
) -> MembershipSnapshot:
    """Strictly reconstruct a checksummed snapshot from a control payload."""

    if set(payload) != {"nodes", "reasons", "snapshot_hash"}:
        raise ValueError("membership payload fields are invalid")
    nodes_value = payload["nodes"]
    reasons_value = payload["reasons"]
    if not isinstance(nodes_value, list) or not isinstance(reasons_value, list):
        raise ValueError("membership nodes and reasons must be lists")
    if type(payload["snapshot_hash"]) is not str:
        raise ValueError("membership snapshot_hash must be a string")
    nodes = []
    for item in nodes_value:
        if not isinstance(item, dict) or set(item) != {
            "agent_id",
            "incarnation_id",
            "addresses",
            "gpu_ids",
        }:
            raise ValueError("membership node fields are invalid")
        addresses = item["addresses"]
        gpu_ids = item["gpu_ids"]
        if (
            type(item["agent_id"]) is not str
            or type(item["incarnation_id"]) is not str
            or not isinstance(addresses, (list, tuple))
            or not isinstance(gpu_ids, (list, tuple))
            or not addresses
            or not gpu_ids
            or not all(type(value) is str for value in (*addresses, *gpu_ids))
        ):
            raise ValueError("membership addresses and gpu_ids must be sequences")
        nodes.append(
            NodeIdentity(
                item["agent_id"],
                item["incarnation_id"],
                tuple(addresses),
                tuple(gpu_ids),
            )
        )
    if not all(isinstance(reason, str) for reason in reasons_value):
        raise ValueError("membership reasons must be strings")
    return MembershipSnapshot(
        generation,
        tuple(nodes),
        tuple(reasons_value),
        payload["snapshot_hash"],
    )


class MembershipStateMachine:
    """Serializes disconnect, lease, join, replacement, and drain events."""

    def __init__(
        self,
        *,
        lease_timeout_s: float = 5.0,
        max_nodes: int | None = None,
        gpu_ids_per_node: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        self.lease_timeout_s = lease_timeout_s
        if max_nodes is not None and max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if gpu_ids_per_node is not None and gpu_ids_per_node < 1:
            raise ValueError("gpu_ids_per_node must be positive")
        self.max_nodes = max_nodes
        self.gpu_ids_per_node = gpu_ids_per_node
        self.clock = clock
        self.generation = 0
        self._nodes: dict[str, _LiveNode] = {}
        self._pending_reasons: set[str] = set()

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    @property
    def has_pending_generation(self) -> bool:
        return bool(self._pending_reasons)

    def _check_generation(self, generation: int) -> None:
        if generation != self.generation:
            raise StaleGeneration(
                f"message generation {generation} does not match active {self.generation}"
            )

    def register(self, identity: NodeIdentity, sequence_number: int) -> None:
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        existing = self._nodes.get(identity.agent_id)
        if existing is None and self.max_nodes is not None and len(self._nodes) >= self.max_nodes:
            raise ValueError(f"membership exceeds configured max_nodes={self.max_nodes}")
        expected_gpu_ids = self.gpu_ids_per_node
        if expected_gpu_ids is None and self._nodes:
            expected_gpu_ids = len(next(iter(self._nodes.values())).identity.gpu_ids)
        if expected_gpu_ids is not None and len(identity.gpu_ids) != expected_gpu_ids:
            raise ValueError(
                f"node {identity.agent_id!r} contributes {len(identity.gpu_ids)} GPUs; "
                f"the fixed per-node TP width is {expected_gpu_ids}"
            )
        if existing and existing.identity.incarnation_id == identity.incarnation_id:
            if sequence_number <= existing.last_sequence:
                raise StaleSequence("duplicate or out-of-order registration")
            existing.last_sequence = sequence_number
            existing.lease_deadline = self.clock() + self.lease_timeout_s
            return
        self._nodes[identity.agent_id] = _LiveNode(
            identity, sequence_number, self.clock() + self.lease_timeout_s
        )
        self._pending_reasons.add(f"{'replacement' if existing else 'join'}:{identity.agent_id}")

    def acknowledge(
        self,
        agent_id: str,
        incarnation_id: str,
        sequence_number: int,
        generation: int,
    ) -> None:
        """Record a preparation acknowledgement under normal ordering rules."""

        self._check_generation(generation)
        node = self._require_incarnation(agent_id, incarnation_id)
        if sequence_number <= node.last_sequence:
            raise StaleSequence("duplicate or out-of-order generation acknowledgement")
        node.last_sequence = sequence_number
        node.lease_deadline = self.clock() + self.lease_timeout_s

    def heartbeat(
        self,
        agent_id: str,
        incarnation_id: str,
        sequence_number: int,
        generation: int,
    ) -> None:
        self._check_generation(generation)
        node = self._require_incarnation(agent_id, incarnation_id)
        if sequence_number <= node.last_sequence:
            raise StaleSequence("duplicate or out-of-order heartbeat")
        node.last_sequence = sequence_number
        node.lease_deadline = self.clock() + self.lease_timeout_s

    def _require_incarnation(self, agent_id: str, incarnation_id: str) -> _LiveNode:
        node = self._nodes.get(agent_id)
        if node is None or node.identity.incarnation_id != incarnation_id:
            raise IncarnationMismatch(f"connection does not own active incarnation for {agent_id}")
        return node

    def disconnect(self, agent_id: str, incarnation_id: str) -> bool:
        node = self._nodes.get(agent_id)
        if node is None or node.identity.incarnation_id != incarnation_id:
            return False
        del self._nodes[agent_id]
        self._pending_reasons.add(f"disconnect:{agent_id}")
        return True

    def drain(
        self,
        agent_id: str,
        incarnation_id: str,
        sequence_number: int,
        generation: int,
    ) -> None:
        self._check_generation(generation)
        node = self._require_incarnation(agent_id, incarnation_id)
        if sequence_number <= node.last_sequence:
            raise StaleSequence("duplicate or out-of-order drain")
        del self._nodes[agent_id]
        self._pending_reasons.add(f"drain:{agent_id}")

    def expire_leases(self, now: float | None = None) -> tuple[str, ...]:
        current = self.clock() if now is None else now
        expired = tuple(
            sorted(
                agent_id for agent_id, node in self._nodes.items() if node.lease_deadline <= current
            )
        )
        for agent_id in expired:
            del self._nodes[agent_id]
            self._pending_reasons.add(f"lease-expired:{agent_id}")
        return expired

    def publish(self) -> MembershipSnapshot | None:
        if not self._pending_reasons:
            return None
        self.generation += 1
        snapshot = MembershipSnapshot(
            self.generation,
            tuple(self._nodes[agent_id].identity for agent_id in sorted(self._nodes)),
            tuple(sorted(self._pending_reasons)),
        )
        self._pending_reasons.clear()
        return snapshot

    def snapshot(self) -> MembershipSnapshot:
        return MembershipSnapshot(
            self.generation,
            tuple(self._nodes[agent_id].identity for agent_id in sorted(self._nodes)),
            (),
        )
