"""Transport-independent, generation-based membership state machine."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

from oobleck.types import OobleckExecutionPlan


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
    removed_nodes: tuple[NodeIdentity, ...]
    added_nodes: tuple[NodeIdentity, ...]
    snapshot_hash: str = field(default="", compare=False)
    previous_execution_plan: OobleckExecutionPlan | None = None
    detection_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.detection_seconds < 0:
            raise ValueError("detection_seconds must be non-negative")
        for field_name, members in (
            ("nodes", self.nodes),
            ("removed_nodes", self.removed_nodes),
            ("added_nodes", self.added_nodes),
        ):
            keys = [(item.agent_id, item.incarnation_id) for item in members]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field_name} contains duplicate membership identities")
        if len({item.agent_id for item in self.nodes}) != len(self.nodes):
            raise ValueError("target membership node IDs must be unique")
        payload = {
            "generation": self.generation,
            "nodes": [asdict(item) for item in self.nodes],
            "removed_nodes": [asdict(item) for item in self.removed_nodes],
            "added_nodes": [asdict(item) for item in self.added_nodes],
            "detection_seconds": self.detection_seconds,
            "previous_execution_plan": (
                self.previous_execution_plan.to_dict()
                if self.previous_execution_plan is not None
                else None
            ),
        }
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("membership snapshot hash is invalid")
        object.__setattr__(self, "snapshot_hash", expected)


def _parse_nodes(field: str, value: object) -> tuple[NodeIdentity, ...]:
    if not isinstance(value, list):
        raise ValueError(f"membership {field} must be a list")
    nodes = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "agent_id",
            "incarnation_id",
            "addresses",
            "gpu_ids",
        }:
            raise ValueError(f"membership {field} node fields are invalid")
        addresses = item["addresses"]
        gpu_ids = item["gpu_ids"]
        if (
            type(item["agent_id"]) is not str
            or type(item["incarnation_id"]) is not str
            or not isinstance(addresses, (list, tuple))
            or not isinstance(gpu_ids, (list, tuple))
            or not addresses
            or not gpu_ids
            or not all(type(member) is str for member in (*addresses, *gpu_ids))
        ):
            raise ValueError(f"membership {field} identity is invalid")
        nodes.append(
            NodeIdentity(
                item["agent_id"],
                item["incarnation_id"],
                tuple(addresses),
                tuple(gpu_ids),
            )
        )
    return tuple(nodes)


def membership_snapshot_from_payload(
    generation: int, payload: Mapping[str, object]
) -> MembershipSnapshot:
    """Strictly reconstruct a checksummed snapshot from a control payload."""

    if set(payload) != {
        "nodes",
        "removed_nodes",
        "added_nodes",
        "detection_seconds",
        "previous_execution_plan",
        "snapshot_hash",
    }:
        raise ValueError("membership payload fields are invalid")
    if type(payload["snapshot_hash"]) is not str:
        raise ValueError("membership snapshot_hash must be a string")
    detection_seconds = payload["detection_seconds"]
    if (
        not isinstance(detection_seconds, (int, float))
        or isinstance(detection_seconds, bool)
        or detection_seconds < 0
    ):
        raise ValueError("membership detection_seconds must be non-negative")
    previous_value = payload["previous_execution_plan"]
    if previous_value is not None and not isinstance(previous_value, Mapping):
        raise ValueError("previous_execution_plan must be an object or null")
    previous_plan = (
        OobleckExecutionPlan.from_dict(previous_value) if previous_value is not None else None
    )
    return MembershipSnapshot(
        generation,
        _parse_nodes("nodes", payload["nodes"]),
        _parse_nodes("removed_nodes", payload["removed_nodes"]),
        _parse_nodes("added_nodes", payload["added_nodes"]),
        payload["snapshot_hash"],
        previous_plan,
        float(detection_seconds),
    )


def is_pure_addition(
    snapshot: MembershipSnapshot,
    active_plan: OobleckExecutionPlan | None,
) -> bool:
    """Return whether a proposal contains additions and no removals."""

    return active_plan is not None and not snapshot.removed_nodes and bool(snapshot.added_nodes)


class MembershipStateMachine:
    """Serializes incarnation-qualified membership removals and additions."""

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
        self._nodes: dict[tuple[str, str], _LiveNode] = {}
        self._pending_removed: dict[tuple[str, str], NodeIdentity] = {}
        self._pending_added: dict[tuple[str, str], NodeIdentity] = {}
        self._pending_detection_seconds = 0.0

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(sorted(node.identity.agent_id for node in self._nodes.values()))

    @property
    def has_pending_generation(self) -> bool:
        return bool(self._pending_removed or self._pending_added)

    def _check_generation(self, generation: int) -> None:
        if generation != self.generation:
            raise StaleGeneration(
                f"message generation {generation} does not match active {self.generation}"
            )

    def register(self, identity: NodeIdentity, sequence_number: int) -> None:
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        existing = next(
            (node for node in self._nodes.values() if node.identity.agent_id == identity.agent_id),
            None,
        )
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
        if existing is not None:
            old = existing.identity
            self._pending_removed[(old.agent_id, old.incarnation_id)] = old
            del self._nodes[(old.agent_id, old.incarnation_id)]
        self._nodes[(identity.agent_id, identity.incarnation_id)] = _LiveNode(
            identity, sequence_number, self.clock() + self.lease_timeout_s
        )
        self._pending_added[(identity.agent_id, identity.incarnation_id)] = identity

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
        node = self._nodes.get((agent_id, incarnation_id))
        if node is None:
            raise IncarnationMismatch(f"connection does not own active incarnation for {agent_id}")
        return node

    def disconnect(self, agent_id: str, incarnation_id: str) -> bool:
        node = self._nodes.get((agent_id, incarnation_id))
        if node is None:
            return False
        identity = node.identity
        del self._nodes[(agent_id, incarnation_id)]
        self._pending_removed[(identity.agent_id, identity.incarnation_id)] = identity
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
        identity = node.identity
        del self._nodes[(agent_id, incarnation_id)]
        self._pending_removed[(identity.agent_id, identity.incarnation_id)] = identity

    def expire_leases(self, now: float | None = None) -> tuple[str, ...]:
        current = self.clock() if now is None else now
        expired = tuple(
            sorted(
                node.identity.agent_id
                for node in self._nodes.values()
                if node.lease_deadline <= current
            )
        )
        for agent_id in expired:
            key, node = next(
                item for item in self._nodes.items() if item[0][0] == agent_id
            )
            identity = node.identity
            del self._nodes[key]
            self._pending_removed[(identity.agent_id, identity.incarnation_id)] = identity
        if expired:
            self._pending_detection_seconds = max(
                self._pending_detection_seconds, self.lease_timeout_s
            )
        return expired

    def publish(self) -> MembershipSnapshot | None:
        if not self.has_pending_generation:
            return None
        self.generation += 1
        snapshot = MembershipSnapshot(
            self.generation,
            tuple(
                sorted(
                    (node.identity for node in self._nodes.values()),
                    key=lambda item: (item.agent_id, item.incarnation_id),
                )
            ),
            tuple(
                sorted(
                    self._pending_removed.values(),
                    key=lambda item: (item.agent_id, item.incarnation_id),
                )
            ),
            tuple(
                sorted(
                    self._pending_added.values(),
                    key=lambda item: (item.agent_id, item.incarnation_id),
                )
            ),
            detection_seconds=self._pending_detection_seconds,
        )
        self._pending_removed.clear()
        self._pending_added.clear()
        self._pending_detection_seconds = 0.0
        return snapshot

    def snapshot(self) -> MembershipSnapshot:
        return MembershipSnapshot(
            self.generation,
            tuple(
                sorted(
                    (node.identity for node in self._nodes.values()),
                    key=lambda item: (item.agent_id, item.incarnation_id),
                )
            ),
            (),
            (),
        )
