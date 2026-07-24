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


def test_rust_and_python_backends_return_identical_templates():
    layers = _profile((91, 45, 29, 75, 73, 66, 99, 81, 83, 83))
    rust = create_pipeline_templates("counterexample", layers, [1, 2, 3, 4, 5, 6])
    python = _create_python_templates("counterexample", layers, [1, 2, 3, 4, 5, 6], 1, None, None)

    assert rust == python
    assert rust[6].layer_ranges == (
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 7),
        (7, 9),
        (9, 10),
    )
    assert rust[6].forward_time + rust[6].backward_time == 164


def test_memory_feasibility_is_part_of_partition_selection():
    layers = _profile((1, 5, 6), (8, 5, 1))
    rust = create_pipeline_templates(
        "memory", layers, [2], device_memory_bytes=10
    )
    python = _create_python_templates("memory", layers, [2], 1, None, 10)

    assert rust == python
    assert rust[2].layer_ranges == ((0, 1), (1, 3))
    assert rust[2].max_microbatches == 1


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
