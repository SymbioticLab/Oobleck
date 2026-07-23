"""Public, serializable value types for the refactored Oobleck runtime.

This module intentionally has no Cornstarch or ``torch.distributed`` imports.  A
generation can therefore be validated and rank-local ownership can be compiled
before WORLD exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


class RecoveryUnavailable(RuntimeError):
    """The available resources cannot form a valid resilient configuration."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def checksum(value: Any) -> str:
    """Return a stable SHA-256 checksum for a JSON-compatible value."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CompatibilityFingerprint:
    model: str
    dtype: str
    tensor_parallel_size: int
    hardware: str
    cornstarch_version: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported template schema version {self.schema_version}; expected 1"
            )

    @property
    def digest(self) -> str:
        return checksum(asdict(self))


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    """Exact software, model, input, and hardware contract for one job."""

    oobleck_commit: str
    cornstarch_commit: str
    torch_version: str
    cuda_version: str | None
    nccl_version: str | None
    model_fingerprint: str
    template_schema_version: int
    optimizer_schema_version: int
    datasets_version: str
    dataset_fingerprint: str
    hardware_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        required = (
            self.oobleck_commit,
            self.cornstarch_commit,
            self.torch_version,
            self.model_fingerprint,
            self.datasets_version,
            self.dataset_fingerprint,
            self.hardware_fingerprint,
        )
        if any(not value for value in required):
            raise ValueError("runtime compatibility fields must not be empty")
        if self.template_schema_version < 1 or self.optimizer_schema_version < 1:
            raise ValueError("schema versions must be positive")
        if self.schema_version != 1:
            raise ValueError(f"unsupported runtime compatibility schema {self.schema_version}")

    @property
    def digest(self) -> str:
        return checksum(asdict(self))

    def assert_matches(self, other: "RuntimeCompatibility") -> None:
        if self != other:
            left = asdict(self)
            right = asdict(other)
            mismatches = [key for key in left if left[key] != right[key]]
            raise ValueError(
                "runtime compatibility mismatch in "
                f"{', '.join(mismatches)} (local={self.digest}, plan={other.digest})"
            )


@dataclass(frozen=True, slots=True)
class PipelineStageSpec:
    """Global layer ownership for one pipeline stage.

    ``layer_start`` is inclusive and ``layer_end`` is exclusive.  Rank order is
    TP-lane order and remains meaningful even before process groups exist.
    """

    pipeline_id: str
    stage_id: int
    layer_start: int
    layer_end: int
    ranks: tuple[int, ...]
    is_first: bool = False
    is_last: bool = False
    tied_parameter_owner: str | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("pipeline_id must not be empty")
        if self.stage_id < 0:
            raise ValueError("stage_id must be >= 0")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("stage layer range must be non-empty and increasing")
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must be a non-empty sequence of unique ranks")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must be non-negative")


@dataclass(frozen=True, slots=True)
class PipelineTemplate:
    """A profiled pipeline layout independent of concrete nodes and ranks."""

    template_id: str
    layer_ranges: tuple[tuple[int, int], ...]
    tensor_parallel_size: int
    forward_time: float
    backward_time: float
    communication_time: float = 0.0
    activation_memory: int = 0
    persistent_memory: int = 0
    max_microbatches: int | None = None
    fingerprint: CompatibilityFingerprint | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("template_id must not be empty")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported template schema version {self.schema_version}; expected 1"
            )
        if not self.layer_ranges:
            raise ValueError("a pipeline template must contain at least one stage")
        previous_end: int | None = None
        for start, end in self.layer_ranges:
            if start < 0 or end <= start:
                raise ValueError("layer ranges must be non-empty and increasing")
            if previous_end is not None and start != previous_end:
                raise ValueError("layer ranges must be contiguous and ordered")
            previous_end = end
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        for name in ("forward_time", "backward_time", "communication_time"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.activation_memory < 0 or self.persistent_memory < 0:
            raise ValueError("memory requirements must be non-negative")
        if self.max_microbatches is not None and self.max_microbatches < 1:
            raise ValueError("max_microbatches must be >= 1 when supplied")

    @property
    def num_stages(self) -> int:
        return len(self.layer_ranges)

    @property
    def resource_count(self) -> int:
        """Number of homogeneous nodes required by this template."""

        return self.num_stages

    def iteration_time(self, microbatches: int) -> float:
        if microbatches < 0:
            raise ValueError("microbatches must be non-negative")
        if self.max_microbatches is not None and microbatches > self.max_microbatches:
            return math.inf
        if microbatches == 0:
            return 0.0
        # Flush time plus steady-state work.  This deliberately uses only the
        # serialized profile and is deterministic across workers.
        stage_work = self.forward_time + self.backward_time
        bubble = max(0, self.num_stages - 1) * stage_work
        return bubble + microbatches * stage_work + self.communication_time

    def assert_compatible(self, fingerprint: CompatibilityFingerprint) -> None:
        if self.fingerprint is None:
            raise ValueError(f"Template {self.template_id!r} has no compatibility fingerprint")
        if self.fingerprint != fingerprint:
            raise ValueError(
                "Pipeline template fingerprint mismatch: "
                f"cached={self.fingerprint.digest}, requested={fingerprint.digest}. "
                "Regenerate the profile/template cache."
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_ranges"] = [list(item) for item in self.layer_ranges]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineTemplate":
        data = dict(value)
        data["layer_ranges"] = tuple(tuple(item) for item in data["layer_ranges"])
        fingerprint = data.get("fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, CompatibilityFingerprint):
            data["fingerprint"] = CompatibilityFingerprint(**fingerprint)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PipelineInstance:
    instance_id: str
    template: PipelineTemplate
    node_ids: tuple[str, ...]
    ranks: tuple[tuple[int, ...], ...] = ()
    microbatches: int = 0

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if len(self.node_ids) != self.template.num_stages:
            raise ValueError("node count must equal the selected template's stage count")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("a node may appear only once in a pipeline")
        if self.ranks and len(self.ranks) != self.template.num_stages:
            raise ValueError("rank groups must contain one entry per stage")
        if any(len(group) != self.template.tensor_parallel_size for group in self.ranks):
            raise ValueError("every stage rank group must match tensor_parallel_size")
        if self.microbatches < 0:
            raise ValueError("microbatches must be non-negative")

    @property
    def stage_specs(self) -> tuple[PipelineStageSpec, ...]:
        if not self.ranks:
            return ()
        return tuple(
            PipelineStageSpec(
                pipeline_id=self.instance_id,
                stage_id=index,
                layer_start=layer_range[0],
                layer_end=layer_range[1],
                ranks=self.ranks[index],
                is_first=index == 0,
                is_last=index == self.template.num_stages - 1,
            )
            for index, layer_range in enumerate(self.template.layer_ranges)
        )


@dataclass(frozen=True, slots=True)
class OobleckExecutionPlan:
    """Checksummed, immutable topology for one membership generation."""

    generation: int
    instances: tuple[PipelineInstance, ...]
    rank_map: tuple[tuple[str, tuple[int, ...]], ...]
    previous_generation: int | None = None
    compatibility_digest: str | None = None
    plan_checksum: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.previous_generation is not None and self.previous_generation >= self.generation:
            raise ValueError("previous_generation must be smaller than generation")
        node_ids = [node for node, _ in self.rank_map]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("rank_map node IDs must be unique")
        ranks = [rank for _, group in self.rank_map for rank in group]
        if len(ranks) != len(set(ranks)) or any(rank < 0 for rank in ranks):
            raise ValueError("rank_map must assign every non-negative rank once")
        instance_nodes = [node for instance in self.instances for node in instance.node_ids]
        if sorted(instance_nodes) != sorted(node_ids):
            raise ValueError("pipeline instances must assign every rank-map node once")
        expected = checksum(self._unsigned_dict())
        if self.plan_checksum and self.plan_checksum != expected:
            raise ValueError("execution plan checksum is invalid")
        object.__setattr__(self, "plan_checksum", expected)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "previous_generation": self.previous_generation,
            "compatibility_digest": self.compatibility_digest,
            "rank_map": [[node, list(ranks)] for node, ranks in self.rank_map],
            "instances": [
                {
                    "instance_id": item.instance_id,
                    "template": item.template.to_dict(),
                    "node_ids": list(item.node_ids),
                    "ranks": [list(group) for group in item.ranks],
                    "microbatches": item.microbatches,
                }
                for item in self.instances
            ],
        }

    def rank_local_stage(self, rank: int) -> PipelineStageSpec:
        for instance in self.instances:
            for spec in instance.stage_specs:
                if rank in spec.ranks:
                    return spec
        raise KeyError(f"rank {rank} is not assigned to this generation")


@dataclass(frozen=True, slots=True)
class OobleckConfig:
    global_batch_size: int
    microbatch_size: int
    fault_tolerance_threshold: int = 0
    max_nodes: int = 1
    seed: int = 0
    heartbeat_interval_s: float = 1.0
    lease_timeout_s: float = 5.0
    rendezvous_port: int = 29500
    rendezvous_timeout_s: float = 60.0
    distributed_backend: str = "auto"
    max_control_frame_bytes: int = 1 << 20
    state_transfer_chunk_bytes: int = 64 << 20
    state_transfer_round_bytes: int = 256 << 20
    transfer_alignment_bytes: int = 256

    def __post_init__(self) -> None:
        if self.global_batch_size < 1 or self.microbatch_size < 1:
            raise ValueError("global_batch_size and microbatch_size must be >= 1")
        if self.global_batch_size % self.microbatch_size:
            raise ValueError("global_batch_size must be divisible by microbatch_size")
        if self.fault_tolerance_threshold < 0:
            raise ValueError("fault_tolerance_threshold must be non-negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        if self.lease_timeout_s <= self.heartbeat_interval_s:
            raise ValueError("lease_timeout_s must exceed heartbeat_interval_s")
        if not 1 <= self.rendezvous_port <= 65535:
            raise ValueError("rendezvous_port must be between 1 and 65535")
        if self.rendezvous_timeout_s <= 0:
            raise ValueError("rendezvous_timeout_s must be positive")
        if self.distributed_backend not in {"auto", "gloo", "nccl"}:
            raise ValueError("distributed_backend must be auto, gloo, or nccl")
        if self.max_control_frame_bytes < 256:
            raise ValueError("max_control_frame_bytes must be >= 256")
        if self.state_transfer_chunk_bytes < 1:
            raise ValueError("state_transfer_chunk_bytes must be positive")
        if self.state_transfer_round_bytes < self.state_transfer_chunk_bytes:
            raise ValueError(
                "state_transfer_round_bytes must be at least state_transfer_chunk_bytes"
            )
        if self.transfer_alignment_bytes < 1:
            raise ValueError("transfer_alignment_bytes must be positive")

    @property
    def global_num_microbatches(self) -> int:
        return self.global_batch_size // self.microbatch_size


def stable_rank_map(
    node_ids: Sequence[str], tensor_parallel_size: int
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    return tuple(
        (
            node,
            tuple(range(index * tensor_parallel_size, (index + 1) * tensor_parallel_size)),
        )
        for index, node in enumerate(sorted(node_ids))
    )
