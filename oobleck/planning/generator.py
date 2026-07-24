"""Paper-traceable Python planner API with equivalent Rust and Python backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from oobleck.planning.profiler import LayerExecutionResult
from oobleck.types import CompatibilityFingerprint, PipelineTemplate


def _effective_activation_memory(layer: LayerExecutionResult) -> int:
    """Read split activation memory, falling back to legacy aggregate profiles."""
    if layer.activation_memory or layer.persistent_memory:
        return layer.activation_memory
    return layer.mem_required


def _sequential_sum(values: Sequence[float]) -> float:
    """Match Rust's left-associated f64 accumulation (Python 3.12 sum is compensated)."""

    result = 0.0
    for value in values:
        result += value
    return result


def _prefix(values: Sequence[float | int]) -> list[float | int]:
    """Build checked prefix sums used for constant-time interval metrics."""
    result: list[float | int] = [0]
    for value in values:
        next_value = result[-1] + value
        if isinstance(next_value, float) and not math.isfinite(next_value):
            raise ValueError("aggregate profile timing overflows float")
        result.append(next_value)
    return result


@dataclass(frozen=True, slots=True)
class _StageMetrics:
    """Aggregate latency and memory for one contiguous pipeline stage."""

    forward: float
    backward: float
    activation: int
    persistent: int

    @property
    def latency(self) -> float:
        """Return the stage's forward-plus-backward compute time."""
        return self.forward + self.backward


@dataclass(frozen=True, slots=True)
class _BottleneckCandidate:
    """A feasible rightmost bottleneck and bounds for stages around it."""

    start: int
    end: int
    latency: float
    min_prefix_stages: int
    min_suffix_stages: int


@dataclass(frozen=True, slots=True)
class _PaperPlan:
    """Internal representation of a partition and its paper objective terms."""

    ranges: tuple[tuple[int, int], ...]
    forward: float
    backward: float
    activation: int
    persistent: int
    capacity: int | None
    t1: float
    t3: float
    kstar: int

    @property
    def planning_iteration_time(self) -> float:
        """Evaluate the Section 4.1.2 iteration-time objective."""
        stages = len(self.ranges)
        return self.t1 + (3 * stages + self.kstar - 1) * (self.forward + self.backward) + self.t3


class _PaperWorkspace:
    """Shared Section 4.1.2 stage cache for all requested templates."""

    def __init__(self, layers: Sequence[LayerExecutionResult], device_memory_bytes: int | None):
        """Cache profile prefixes shared by every requested stage count."""
        self.layers = layers
        self.device_memory_bytes = device_memory_bytes
        self.forward = _prefix([layer.forward for layer in layers])
        self.backward = _prefix([layer.backward for layer in layers])
        self.activation = _prefix([_effective_activation_memory(layer) for layer in layers])
        self.persistent = _prefix([layer.persistent_memory for layer in layers])

    def stage(self, start: int, end: int) -> _StageMetrics:
        """Aggregate timing and memory for the half-open layer interval."""
        return _StageMetrics(
            float(self.forward[end] - self.forward[start]),
            float(self.backward[end] - self.backward[start]),
            int(self.activation[end] - self.activation[start]),
            int(self.persistent[end] - self.persistent[start]),
        )

    def allowed(self, start: int, end: int, threshold: float, *, inclusive: bool) -> bool:
        """Test both device-memory feasibility and a bottleneck latency bound."""
        stage = self.stage(start, end)
        if (
            self.device_memory_bytes is not None
            and stage.activation + stage.persistent > self.device_memory_bytes
        ):
            return False
        return stage.latency <= threshold if inclusive else stage.latency < threshold

    def min_segments(
        self, start: int, end: int, threshold: float, *, inclusive: bool
    ) -> int | None:
        """Greedily conquer an interval with the fewest bounded stages."""

        position = start
        count = 0
        while position < end:
            farthest = None
            for next_position in range(position + 1, end + 1):
                if self.allowed(position, next_position, threshold, inclusive=inclusive):
                    farthest = next_position
                else:
                    break
            if farthest is None:
                return None
            position = farthest
            count += 1
        return count

    def lexicographic_partition(
        self,
        start: int,
        end: int,
        count: int,
        threshold: float,
        *,
        inclusive: bool,
    ) -> tuple[int, ...] | None:
        """Find the lexicographically earliest feasible contiguous partition.

        Each next boundary is chosen as far left as possible while a greedy
        minimum-segment check proves that the suffix can still use exactly the
        remaining number of stages. This deterministic tie-break is shared with
        the Rust backend and keeps equal-cost plans byte-for-byte reproducible.
        """
        if count == 0:
            return () if start == end else None
        if count > end - start:
            return None

        starts: list[int] = []
        position = start
        for stage_index in range(count):
            starts.append(position)
            remaining = count - stage_index - 1
            if remaining == 0:
                return (
                    tuple(starts)
                    if self.allowed(position, end, threshold, inclusive=inclusive)
                    else None
                )
            latest = end - remaining
            chosen = None
            for next_position in range(position + 1, latest + 1):
                if not self.allowed(position, next_position, threshold, inclusive=inclusive):
                    break
                minimum = self.min_segments(next_position, end, threshold, inclusive=inclusive)
                if minimum is not None and minimum <= remaining <= end - next_position:
                    chosen = next_position
                    break
            if chosen is None:
                return None
            position = chosen
        return None

    def bottleneck_candidates(self) -> tuple[_BottleneckCandidate, ...]:
        """Enumerate feasible intervals that can be the rightmost bottleneck.

        Prefix stages may tie the candidate latency, but suffix stages must be
        strictly faster. This encodes the paper's ``k*`` convention and avoids
        emitting duplicate interpretations of a partition with tied maxima.
        """
        candidates = []
        layer_count = len(self.layers)
        for start in range(layer_count):
            for end in range(start + 1, layer_count + 1):
                stage = self.stage(start, end)
                if (
                    self.device_memory_bytes is not None
                    and stage.activation + stage.persistent > self.device_memory_bytes
                ):
                    continue
                prefix = self.min_segments(0, start, stage.latency, inclusive=True)
                # k* is the rightmost bottleneck: following stages must be
                # strictly faster, while preceding stages may tie it.
                suffix = self.min_segments(end, layer_count, stage.latency, inclusive=False)
                if prefix is not None and suffix is not None:
                    candidates.append(
                        _BottleneckCandidate(start, end, stage.latency, prefix, suffix)
                    )
        return tuple(candidates)

    def materialize(
        self, candidate: _BottleneckCandidate, prefix_stages: int, total_stages: int
    ) -> _PaperPlan | None:
        """Construct and score a complete plan around one bottleneck.

        Prefix and suffix partitions obey their respective inclusive/strict
        latency bounds. The resulting plan records maximum per-stage memory,
        the largest feasible microbatch count, and the T1/T3 objective terms
        needed to compare it with other candidates.
        """
        suffix_stages = total_stages - prefix_stages - 1
        prefix = self.lexicographic_partition(
            0,
            candidate.start,
            prefix_stages,
            candidate.latency,
            inclusive=True,
        )
        suffix = self.lexicographic_partition(
            candidate.end,
            len(self.layers),
            suffix_stages,
            candidate.latency,
            inclusive=False,
        )
        if prefix is None or suffix is None:
            return None
        starts = (*prefix, candidate.start, *suffix)
        ends = (*starts[1:], len(self.layers))
        ranges = tuple(zip(starts, ends))
        metrics = tuple(self.stage(start, end) for start, end in ranges)
        bottleneck = metrics[prefix_stages]
        stage_memory = tuple((item.activation, item.persistent) for item in metrics)
        return _PaperPlan(
            ranges,
            bottleneck.forward,
            bottleneck.backward,
            max(item.activation for item in metrics),
            max(item.persistent for item in metrics),
            _max_microbatches(stage_memory, self.device_memory_bytes),
            _sequential_sum(tuple(item.latency for item in metrics)),
            _sequential_sum(tuple(item.latency for item in metrics[prefix_stages:])),
            prefix_stages,
        )


def _paper_plans(
    layers: Sequence[LayerExecutionResult],
    resource_counts: Sequence[int],
    device_memory_bytes: int | None = None,
) -> dict[int, _PaperPlan]:
    """Apply Section 4.1.2 Equations 1--4 for fixed-TP logical nodes.

    The refactored runtime assigns one complete fixed-TP node to each stage, so
    the paper state T(S, u, v, d) has S=d and no within-node GPU split m. The
    shared bottleneck cache is equivalent to evaluating every feasible divide
    and conquer result, but retains enough state to avoid the artifact's unsafe
    single-locally-best-subproblem assumption.
    """

    requested = sorted(set(resource_counts))
    if not requested or requested[0] < 1 or requested[-1] > len(layers):
        raise ValueError("stage count must be between one and the number of layers")
    workspace = _PaperWorkspace(layers, device_memory_bytes)
    candidates = workspace.bottleneck_candidates()
    results: dict[int, _PaperPlan] = {}

    for stages in requested:
        best: _PaperPlan | None = None
        for candidate in candidates:
            prefix_lower = max(
                candidate.min_prefix_stages,
                stages - 1 - (len(layers) - candidate.end),
                0,
            )
            prefix_upper = min(
                candidate.start,
                stages - 1 - candidate.min_suffix_stages,
            )
            if prefix_lower > prefix_upper or prefix_lower >= stages:
                continue
            prefix_counts = (
                range(prefix_lower, prefix_upper + 1)
                if candidate.latency == 0.0
                else (prefix_lower,)
            )
            for prefix_stages in prefix_counts:
                predicted = (
                    workspace.stage(0, len(layers)).latency
                    + (3 * stages + prefix_stages - 1) * candidate.latency
                    + workspace.stage(candidate.start, len(layers)).latency
                )
                if best is not None and predicted > best.planning_iteration_time:
                    continue
                plan = workspace.materialize(candidate, prefix_stages, stages)
                if plan is not None and (
                    best is None
                    or (plan.planning_iteration_time, tuple(start for start, _ in plan.ranges))
                    < (
                        best.planning_iteration_time,
                        tuple(start for start, _ in best.ranges),
                    )
                ):
                    best = plan
        if best is None:
            raise ValueError(
                f"pipeline template for resource count {stages} cannot fit one microbatch in device memory"
            )
        results[stages] = best
    return results


def _partitions(
    layers: Sequence[LayerExecutionResult],
    resource_counts: Sequence[int],
    device_memory_bytes: int | None = None,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Compatibility helper returning only the paper planner's layer ranges."""

    return {
        stages: plan.ranges
        for stages, plan in _paper_plans(layers, resource_counts, device_memory_bytes).items()
    }


def _partition(
    layers: Sequence[LayerExecutionResult],
    stages: int,
    device_memory_bytes: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return the paper planner's ranges for one stage count."""
    return _partitions(layers, (stages,), device_memory_bytes)[stages]


def _max_microbatches(
    stage_memory: Sequence[tuple[int, int]], device_memory_bytes: int | None
) -> int | None:
    """Compute the tightest per-stage activation capacity after persistent state."""
    if device_memory_bytes is None:
        return None
    capacities = []
    for activation_memory, persistent_memory in stage_memory:
        available = device_memory_bytes - persistent_memory
        if available < 0 or (activation_memory > 0 and available < activation_memory):
            raise ValueError("pipeline template cannot fit one microbatch in device memory")
        if activation_memory > 0:
            capacities.append(available // activation_memory)
    return min(capacities) if capacities else None


def _create_python_templates(
    model_name: str,
    profile_data: Sequence[LayerExecutionResult],
    resource_counts: Sequence[int],
    tensor_parallel_size: int,
    fingerprint: CompatibilityFingerprint | None,
    device_memory_bytes: int | None,
) -> dict[int, PipelineTemplate]:
    """Materialize public templates from the pure-Python paper planner."""
    results = {}
    for stages, plan in _paper_plans(profile_data, resource_counts, device_memory_bytes).items():
        results[stages] = PipelineTemplate(
            f"{model_name}-stages-{stages}",
            plan.ranges,
            tensor_parallel_size,
            plan.forward,
            plan.backward,
            activation_memory=plan.activation,
            persistent_memory=plan.persistent,
            max_microbatches=plan.capacity,
            fingerprint=fingerprint,
            paper_t1=plan.t1,
            paper_t3=plan.t3,
            paper_bottleneck_stage=plan.kstar,
        )
    return results


def create_pipeline_templates(
    model_name: str,
    profile_data: Sequence[LayerExecutionResult],
    num_nodes: Sequence[int],
    tensor_parallel_size: int = 1,
    *,
    fingerprint: CompatibilityFingerprint | None = None,
    device_memory_bytes: int | None = None,
) -> dict[int, PipelineTemplate]:
    """Create a paper-modeled fixed-TP template for every requested node count.

    Inputs are validated once, including contiguous global layer indices and
    optional device capacity. The native Rust backend is preferred when the
    extension is installed; the Python implementation is the behavioral
    fallback. Both enumerate the Section 4.1.2 bottleneck candidates, apply the
    same deterministic tie-breaking, and expose identical T1/T3/k* metadata.

    The fixed tensor-parallel width makes each logical node one pipeline stage.
    Infeasible memory constraints are normalized to ``ValueError`` so callers do
    not depend on the selected backend's internal error type.
    """

    if not model_name or not profile_data or not num_nodes:
        raise ValueError("model_name, profile_data, and num_nodes must be non-empty")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if any(count < 1 for count in num_nodes):
        raise ValueError("num_nodes must contain only positive resource counts")
    if device_memory_bytes is not None and device_memory_bytes < 1:
        raise ValueError("device_memory_bytes must be positive when supplied")
    if tuple(layer.layer_index for layer in profile_data) != tuple(range(len(profile_data))):
        raise ValueError("profile_data must have contiguous global layer indices")
    try:
        from oobleck.planning import planner as rust_planner
    except ImportError:
        rust_planner = None
    if rust_planner is None:
        return _create_python_templates(
            model_name,
            profile_data,
            num_nodes,
            tensor_parallel_size,
            fingerprint,
            device_memory_bytes,
        )

    try:
        raw = rust_planner.create_pipeline_templates(
            model_name,
            list(profile_data),
            list(num_nodes),
            tensor_parallel_size,
            device_memory_bytes,
        )
    except RuntimeError as exc:
        # Keep the supported Python API backend-independent while the direct
        # extension preserves an explicit infeasible-planning error category.
        raise ValueError(str(exc)) from exc
    return {
        int(stages): PipelineTemplate(
            value["template_id"],
            tuple(tuple(item) for item in value["layer_ranges"]),
            int(value["tensor_parallel_size"]),
            float(value["forward_time"]),
            float(value["backward_time"]),
            float(value["communication_time"]),
            int(value["activation_memory"]),
            int(value["persistent_memory"]),
            None if value["max_microbatches"] is None else int(value["max_microbatches"]),
            fingerprint,
            int(value["schema_version"]),
            float(value["paper_t1"]),
            float(value["paper_t3"]),
            int(value["paper_bottleneck_stage"]),
        )
        for stages, value in raw.items()
    }
