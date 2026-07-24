"""Narrow Python planner API with equivalent Rust and Python backends."""

from __future__ import annotations

import math
from typing import Sequence

from oobleck.planning.profiler import LayerExecutionResult
from oobleck.types import CompatibilityFingerprint, PipelineTemplate


def _effective_activation_memory(layer: LayerExecutionResult) -> int:
    if layer.activation_memory or layer.persistent_memory:
        return layer.activation_memory
    return layer.mem_required


def _prefix(values: Sequence[float | int]) -> list[float | int]:
    result: list[float | int] = [0]
    for value in values:
        next_value = result[-1] + value
        if isinstance(next_value, float) and not math.isfinite(next_value):
            raise ValueError("aggregate profile timing overflows float")
        result.append(next_value)
    return result


def _partitions(
    layers: Sequence[LayerExecutionResult],
    resource_counts: Sequence[int],
    device_memory_bytes: int | None = None,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Return exact deterministic minimax partitions from one dynamic program."""

    requested = sorted(set(resource_counts))
    if not requested or requested[0] < 1 or requested[-1] > len(layers):
        raise ValueError("stage count must be between one and the number of layers")
    forward = _prefix([layer.forward for layer in layers])
    backward = _prefix([layer.backward for layer in layers])
    activation = _prefix([_effective_activation_memory(layer) for layer in layers])
    persistent = _prefix([layer.persistent_memory for layer in layers])

    # dp[k][j] = (maximum stage work, stage start positions) for k stages
    # covering the first j layers. Keeping the lexicographically smallest
    # starts gives a stable answer when several partitions have equal cost.
    max_stages = requested[-1]
    dp: list[list[tuple[float, tuple[int, ...]] | None]] = [
        [None] * (len(layers) + 1) for _ in range(max_stages + 1)
    ]
    dp[0][0] = (0.0, ())
    for count in range(1, max_stages + 1):
        for end in range(count, len(layers) + 1):
            candidates = []
            for start in range(count - 1, end):
                previous = dp[count - 1][start]
                if previous is None:
                    continue
                stage_activation = int(activation[end] - activation[start])
                stage_persistent = int(persistent[end] - persistent[start])
                if (
                    device_memory_bytes is not None
                    and stage_activation + stage_persistent > device_memory_bytes
                ):
                    continue
                stage_work = float(
                    forward[end] - forward[start] + backward[end] - backward[start]
                )
                candidates.append(
                    (max(previous[0], stage_work), (*previous[1], start))
                )
            if candidates:
                dp[count][end] = min(candidates)

    results = {}
    for stages in requested:
        result = dp[stages][len(layers)]
        if result is None:
            raise ValueError(
                f"pipeline template for resource count {stages} cannot fit one microbatch in device memory"
            )
        starts = result[1]
        ends = (*starts[1:], len(layers))
        results[stages] = tuple(
            (layers[start].layer_index, layers[end - 1].layer_index + 1)
            for start, end in zip(starts, ends)
        )
    return results


def _partition(
    layers: Sequence[LayerExecutionResult],
    stages: int,
    device_memory_bytes: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return one partition; retained as the small Python oracle entry point."""

    return _partitions(layers, (stages,), device_memory_bytes)[stages]


def _max_microbatches(
    stage_memory: Sequence[tuple[int, int]], device_memory_bytes: int | None
) -> int | None:
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


def _summarize_partition(
    layers: Sequence[LayerExecutionResult],
    ranges: Sequence[tuple[int, int]],
    device_memory_bytes: int | None,
) -> tuple[float, float, int, int, int | None]:
    forward_prefix = _prefix([layer.forward for layer in layers])
    backward_prefix = _prefix([layer.backward for layer in layers])
    activation_prefix = _prefix([_effective_activation_memory(layer) for layer in layers])
    persistent_prefix = _prefix([layer.persistent_memory for layer in layers])
    stage_metrics = [
        (
            float(forward_prefix[end] - forward_prefix[start]),
            float(backward_prefix[end] - backward_prefix[start]),
            int(activation_prefix[end] - activation_prefix[start]),
            int(persistent_prefix[end] - persistent_prefix[start]),
        )
        for start, end in ranges
    ]
    bottleneck = max(
        range(len(stage_metrics)),
        key=lambda index: (stage_metrics[index][0] + stage_metrics[index][1], index),
    )
    forward, backward, _, _ = stage_metrics[bottleneck]
    activation_memory = max(item[2] for item in stage_metrics)
    persistent_memory = max(item[3] for item in stage_metrics)
    max_microbatches = _max_microbatches(
        [(item[2], item[3]) for item in stage_metrics], device_memory_bytes
    )
    return forward, backward, activation_memory, persistent_memory, max_microbatches


def _create_python_templates(
    model_name: str,
    profile_data: Sequence[LayerExecutionResult],
    resource_counts: Sequence[int],
    tensor_parallel_size: int,
    fingerprint: CompatibilityFingerprint | None,
    device_memory_bytes: int | None,
) -> dict[int, PipelineTemplate]:
    results = {}
    partitions = _partitions(profile_data, resource_counts, device_memory_bytes)
    for stages, ranges in partitions.items():
        forward, backward, activation, persistent, capacity = _summarize_partition(
            profile_data, ranges, device_memory_bytes
        )
        results[stages] = PipelineTemplate(
            f"{model_name}-stages-{stages}",
            ranges,
            tensor_parallel_size,
            forward,
            backward,
            activation_memory=activation,
            persistent_memory=persistent,
            max_microbatches=capacity,
            fingerprint=fingerprint,
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
    """Create one fixed-TP, one-stage-per-resource template for each count."""

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
        )
        for stages, value in raw.items()
    }
