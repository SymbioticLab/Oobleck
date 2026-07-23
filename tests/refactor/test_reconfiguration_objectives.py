from __future__ import annotations

from oobleck.planning import PipelineInstance, PipelineTemplate, reconfigure_pipelines


def template(resources: int, *, speed: float = 1.0) -> PipelineTemplate:
    return PipelineTemplate(
        f"p{resources}-{speed}",
        tuple((index, index + 1) for index in range(resources)),
        1,
        speed,
        speed,
    )


def test_borrow_moves_the_lowest_state_byte_node_after_throughput_tie():
    two, three = template(2), template(3)
    previous = (
        PipelineInstance("a", three, ("a0", "a1", "a2")),
        PipelineInstance("b", three, ("b0", "b1", "b2")),
    )
    result = reconfigure_pipelines(
        previous,
        {"a0", "b0", "b1", "b2"},
        [two, three],
        4,
        1,
        state_bytes_by_node={"a0": 10, "b0": 100, "b1": 1, "b2": 50},
    )
    owner = {node: item.instance_id for item in result.instances for node in item.node_ids}
    assert owner["b1"] == "a"
    assert owner["b0"] == owner["b2"] == "b"
    assert result.retained_state_bytes == 160


def test_merge_keeps_the_identity_with_more_state_bytes():
    two, four = template(2), template(4)
    previous = (
        PipelineInstance("a", two, ("a0", "a1")),
        PipelineInstance("b", two, ("b0", "b1")),
        PipelineInstance("c", two, ("c0", "c1")),
    )
    result = reconfigure_pipelines(
        previous,
        {"a0", "b0", "c0", "c1"},
        [two, four],
        4,
        1,
        state_bytes_by_node={"a0": 1, "b0": 100, "c0": 20, "c1": 20},
    )
    merged = next(item for item in result.instances if {"a0", "b0"} <= set(item.node_ids))
    assert merged.instance_id == "b"
    assert result.retained_state_bytes == 140


def test_simultaneous_failures_across_multiple_pipelines_are_atomic():
    two, three = template(2), template(3)
    previous = tuple(
        PipelineInstance(prefix, three, tuple(f"{prefix}{index}" for index in range(3)))
        for prefix in ("a", "b", "c")
    )
    survivors = {"a0", "a1", "b0", "b1", "c0", "c1", "c2"}
    result = reconfigure_pipelines(previous, survivors, [two, three], 6, 2)
    assert result.strategies == ("simple",)
    assert {node for item in result.instances for node in item.node_ids} == survivors
    assert len(result.instances) == 3
    assert result.moved_nodes == 0


def test_join_preserves_old_pipeline_ownership_when_throughput_is_equal():
    two = template(2)
    previous = (
        PipelineInstance("pipeline-0000", two, ("b", "c")),
        PipelineInstance("pipeline-0001", two, ("d", "e")),
    )
    result = reconfigure_pipelines(
        previous,
        {"a", "b", "c", "d", "e", "f"},
        [two],
        6,
        2,
        state_bytes_by_node={"b": 50, "c": 50, "d": 40, "e": 40},
    )
    owners = {item.instance_id: set(item.node_ids) for item in result.instances}
    assert owners["pipeline-0000"] == {"b", "c"}
    assert owners["pipeline-0001"] == {"d", "e"}
    assert result.retained_nodes == 4
    assert result.retained_state_bytes == 180
