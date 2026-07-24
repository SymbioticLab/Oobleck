from itertools import combinations
import math

import pytest

from oobleck.planning import LayerExecutionResult, create_pipeline_templates
from oobleck.planning.generator import _create_python_templates


def _profile(
    latencies: tuple[float, ...], activation: tuple[int, ...] | None = None
) -> list[LayerExecutionResult]:
    memory = activation or (1,) * len(latencies)
    return [
        LayerExecutionResult(index, f"layer.{index}", latency, 0.0, memory[index], memory[index], 0)
        for index, latency in enumerate(latencies)
    ]


def _paper_oracle(latencies: tuple[float, ...], stages: int) -> tuple[float, tuple[int, ...]]:
    best: tuple[float, tuple[int, ...]] | None = None
    for cuts in combinations(range(1, len(latencies)), stages - 1):
        starts = (0, *cuts)
        ends = (*cuts, len(latencies))
        work = tuple(sum(latencies[start:end]) for start, end in zip(starts, ends))
        kstar = max(range(stages), key=lambda index: (work[index], index))
        t1 = sum(work)
        t2 = (3 * stages + kstar - 1) * work[kstar]
        t3 = sum(work[kstar:])
        candidate = (t1 + t2 + t3, starts)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def test_template_generator_covers_uneven_global_ranges():
    layers = [
        LayerExecutionResult(index, f"layer.{index}", index + 1, index + 1, 10)
        for index in range(6)
    ]
    templates = create_pipeline_templates("toy", layers, [1, 2, 3], 1)
    assert set(templates) == {1, 2, 3}
    for stages, template in templates.items():
        assert template.num_stages == stages
        covered = [index for start, end in template.layer_ranges for index in range(start, end)]
        assert covered == list(range(6))
    assert templates[2].layer_ranges == ((0, 4), (4, 6))


def test_rust_and_python_backends_match_the_paper_objective():
    latencies = (91, 45, 29, 75, 73, 66, 99, 81, 83, 83)
    layers = _profile(latencies)
    rust = create_pipeline_templates("counterexample", layers, [1, 2, 3, 4, 5, 6])
    python = _create_python_templates("counterexample", layers, [1, 2, 3, 4, 5, 6], 1, None, None)

    assert rust == python
    expected_time, expected_starts = _paper_oracle(latencies, 6)
    assert tuple(start for start, _ in rust[6].layer_ranges) == expected_starts
    assert rust[6].planning_iteration_time == expected_time
    assert rust[6].paper_t1 == sum(latencies)
    assert rust[6].paper_t3 is not None
    assert rust[6].paper_bottleneck_stage is not None


def test_memory_feasibility_is_part_of_partition_selection():
    layers = _profile((1, 5, 6), (8, 5, 1))
    rust = create_pipeline_templates("memory", layers, [2], device_memory_bytes=10)
    python = _create_python_templates("memory", layers, [2], 1, None, 10)

    assert rust == python
    assert rust[2].layer_ranges == ((0, 1), (1, 3))
    assert rust[2].max_microbatches == 1
    assert math.isfinite(rust[2].planning_iteration_time)


def test_memory_infeasibility_reports_requested_resource_count():
    layers = _profile((1, 1), (11, 1))
    with pytest.raises(ValueError, match="resource count 1"):
        create_pipeline_templates("memory", layers, [1], device_memory_bytes=10)

    from oobleck.planning import planner as rust_planner

    with pytest.raises(RuntimeError, match="resource count 1"):
        rust_planner.create_pipeline_templates("memory", layers, [1], 1, 10)


def test_profile_timings_must_be_finite():
    with pytest.raises(ValueError, match="non-negative"):
        LayerExecutionResult(0, "bad", float("nan"), 1.0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        LayerExecutionResult(0, "bad", 1.0, float("inf"), 1)
    with pytest.raises(ValueError, match="non-negative"):
        LayerExecutionResult(0, "bad", 1.0e308, 1.0e308, 1)

    overflowing = _profile((1.0e308, 1.0e308))
    with pytest.raises(ValueError, match="overflows"):
        create_pipeline_templates("overflow", overflowing, [1])
    with pytest.raises(ValueError, match="overflows"):
        _create_python_templates("overflow", overflowing, [1], 1, None, None)


def test_resource_counts_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        create_pipeline_templates("toy", _profile((1, 1)), [0])
