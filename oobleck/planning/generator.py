"""Narrow Python planner API with a deterministic fallback for the Rust extension."""

from __future__ import annotations

from typing import Sequence

from oobleck.planning.profiler import LayerExecutionResult
from oobleck.types import CompatibilityFingerprint, PipelineTemplate


def _partition(layers: Sequence[LayerExecutionResult], stages: int) -> tuple[tuple[int, int], ...]:
    if stages < 1 or stages > len(layers):
        raise ValueError("stage count must be between one and the number of layers")
    prefix = [0.0]
    for layer in layers:
        prefix.append(prefix[-1] + layer.forward + layer.backward)
    # dp[k][j] = (maximum stage work, cut positions) for k stages covering j layers.
    dp: list[list[tuple[float, tuple[int, ...]] | None]] = [
        [None] * (len(layers) + 1) for _ in range(stages + 1)
    ]
    dp[0][0] = (0.0, ())
    for count in range(1, stages + 1):
        for end in range(count, len(layers) + 1):
            candidates = []
            for cut in range(count - 1, end):
                previous = dp[count - 1][cut]
                if previous is None:
                    continue
                work = prefix[end] - prefix[cut]
                candidates.append((max(previous[0], work), (*previous[1], cut)))
            dp[count][end] = min(candidates, key=lambda item: (item[0], item[1]))
    result = dp[stages][len(layers)]
    assert result is not None
    cuts = (*result[1], len(layers))
    return tuple(
        (layers[cuts[index]].layer_index, layers[cuts[index + 1] - 1].layer_index + 1)
        for index in range(stages)
    )


def _max_microbatches(
    activation_memory: int, persistent_memory: int, device_memory_bytes: int | None
) -> int | None:
    if device_memory_bytes is None:
        return None
    available = device_memory_bytes - persistent_memory
    if available < 0 or (activation_memory > 0 and available < activation_memory):
        raise ValueError("pipeline template cannot fit one microbatch in device memory")
    if activation_memory == 0:
        return None
    return available // activation_memory


def create_pipeline_templates(
    model_name: str,
    profile_data: Sequence[LayerExecutionResult],
    num_nodes: Sequence[int],
    tensor_parallel_size: int = 1,
    *,
    fingerprint: CompatibilityFingerprint | None = None,
    device_memory_bytes: int | None = None,
) -> dict[int, PipelineTemplate]:
    if not model_name or not profile_data or not num_nodes:
        raise ValueError("model_name, profile_data, and num_nodes must be non-empty")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if device_memory_bytes is not None and device_memory_bytes < 1:
        raise ValueError("device_memory_bytes must be positive when supplied")
    if tuple(layer.layer_index for layer in profile_data) != tuple(range(len(profile_data))):
        raise ValueError("profile_data must have contiguous global layer indices")
    try:
        from oobleck.planning import planner as rust_planner
    except ImportError:
        rust_planner = None
    if rust_planner is not None:
        raw = rust_planner.create_pipeline_templates(
            model_name,
            list(profile_data),
            list(num_nodes),
            tensor_parallel_size,
        )
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
                _max_microbatches(
                    int(value["activation_memory"]),
                    int(value["persistent_memory"]),
                    device_memory_bytes,
                ),
                fingerprint,
                int(value["schema_version"]),
            )
            for stages, value in raw.items()
        }
    results = {}
    for stages in sorted(set(num_nodes)):
        ranges = _partition(profile_data, stages)
        forward = max(
            sum(layer.forward for layer in profile_data[start:end]) for start, end in ranges
        )
        backward = max(
            sum(layer.backward for layer in profile_data[start:end]) for start, end in ranges
        )
        activation_memory = max(
            sum(
                layer.activation_memory
                if layer.activation_memory or layer.persistent_memory
                else layer.mem_required
                for layer in profile_data[start:end]
            )
            for start, end in ranges
        )
        persistent_memory = max(
            sum(layer.persistent_memory for layer in profile_data[start:end])
            for start, end in ranges
        )
        results[stages] = PipelineTemplate(
            f"{model_name}-stages-{stages}",
            ranges,
            tensor_parallel_size,
            forward,
            backward,
            activation_memory=activation_memory,
            persistent_memory=persistent_memory,
            max_microbatches=_max_microbatches(
                activation_memory, persistent_memory, device_memory_bytes
            ),
            fingerprint=fingerprint,
        )
    return results
