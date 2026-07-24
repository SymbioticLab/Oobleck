from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from oobleck import OobleckConfig, OobleckParallelizationPlan, RuntimeCompatibility
from oobleck.planning import (
    CompatibilityFingerprint,
    LayerExecutionResult,
    ModelProfile,
    ModelProfiler,
    PipelineInstance,
    PipelineTemplate,
    ProfilingWorkload,
    RecoveryUnavailable,
    compose_templates,
    create_pipeline_templates,
    load_templates,
    reconfigure_pipelines,
    save_templates,
)


def template(stages: int, *, name: str | None = None, speed: float = 1.0):
    return PipelineTemplate(
        name or f"p{stages}",
        tuple((index, index + 1) for index in range(stages)),
        1,
        speed,
        speed,
    )


def test_template_cache_rejects_incompatible_fingerprint(tmp_path):
    fp = CompatibilityFingerprint("model", "bf16", 1, "gpu", "abc")
    item = PipelineTemplate("p1", ((0, 2),), 1, 1, 1, fingerprint=fp)
    path = tmp_path / "templates.json"
    save_templates(path, [item], fp)
    assert load_templates(path, fp) == (item,)
    other = CompatibilityFingerprint("other", "bf16", 1, "gpu", "abc")
    with pytest.raises(ValueError, match="incompatible"):
        load_templates(path, other)


def test_profiler_measures_and_records_materialized_layers(tmp_path):
    layers = (torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    workload = ProfilingWorkload(
        layers,
        lambda layer_index, iteration: torch.ones(
            2, 3 if layer_index == 0 else 4, requires_grad=True
        ),
    )
    fingerprint = CompatibilityFingerprint("tiny", "float32", 1, "cpu", "test")
    path = tmp_path / "measured.json"
    profile = ModelProfiler("tiny", fingerprint=fingerprint).measure_and_record(
        path, 2, workload, warmup_steps=0, measurement_steps=2
    )
    assert [item.layer_index for item in profile.layers] == [0, 1]
    assert all(item.forward > 0 and item.backward > 0 for item in profile.layers)
    assert all(item.mem_required > 0 for item in profile.layers)
    assert all(item.persistent_memory > 0 for item in profile.layers)
    assert ModelProfile.load(path, fingerprint) == profile
    templates = create_pipeline_templates("tiny", profile.layers, (1, 2), fingerprint=fingerprint)
    assert templates[1].forward_time == pytest.approx(sum(item.forward for item in profile.layers))
    assert templates[1].backward_time == pytest.approx(
        sum(item.backward for item in profile.layers)
    )
    assert all(item.persistent_memory > 0 for item in templates.values())
    assert all(item.activation_memory >= 0 for item in templates.values())


def test_template_capacity_uses_device_memory_budget():
    layers = (
        LayerExecutionResult(0, "l0", 1.0, 1.0, 300, 100, 200),
        LayerExecutionResult(1, "l1", 1.0, 1.0, 300, 100, 200),
    )
    templates = create_pipeline_templates("tiny", layers, (1, 2), device_memory_bytes=1000)
    assert templates[1].max_microbatches == 3
    assert templates[2].max_microbatches == 8
    with pytest.raises(ValueError, match="cannot fit one microbatch"):
        create_pipeline_templates("tiny", layers, (1,), device_memory_bytes=500)


def test_composition_assigns_every_node_and_batch_once():
    instances = compose_templates(
        [template(1), template(2, speed=0.4)],
        ["d", "c", "b", "a"],
        total_microbatches=8,
        fault_tolerance_threshold=1,
    )
    assert sorted(node for item in instances for node in item.node_ids) == ["a", "b", "c", "d"]
    assert sum(item.microbatches for item in instances) == 8
    assert len(instances) >= 2


def test_reconfiguration_simple_borrow_and_merge():
    two, three, four = template(2), template(3), template(4)
    previous = (
        PipelineInstance("a", three, ("a0", "a1", "a2")),
        PipelineInstance("b", three, ("b0", "b1", "b2")),
    )
    simple = reconfigure_pipelines(previous, {"a0", "a1", "b0", "b1", "b2"}, [two, three], 4, 1)
    assert "simple" in simple.strategies

    borrowed = reconfigure_pipelines(previous, {"a0", "b0", "b1", "b2"}, [two, three], 4, 1)
    assert "borrow" in borrowed.strategies
    assert sorted(len(item.node_ids) for item in borrowed.instances) == [2, 2]

    previous_merge = (
        PipelineInstance("a", two, ("a0", "a1")),
        PipelineInstance("b", two, ("b0", "b1")),
        PipelineInstance("c", two, ("c0", "c1")),
    )
    merged = reconfigure_pipelines(previous_merge, {"a0", "b0", "c0", "c1"}, [two, four], 4, 1)
    assert "merge" in merged.strategies
    assert len(merged.instances) == 2


def test_reconfiguration_returns_explicit_unavailable():
    two = template(2)
    previous = (PipelineInstance("a", two, ("a0", "a1")),)
    with pytest.raises(RecoveryUnavailable):
        reconfigure_pipelines(previous, {"a0"}, [two], 2, 0)


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: None = None
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1


def test_execution_plan_tracks_stable_node_when_join_reorders_global_ranks():
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=4, microbatch_size=1, max_nodes=3),
        templates=(template(1),),
        node_ids=("node-b", "node-c"),
        rank=0,
    )
    plan.parallelize(torch.nn.Sequential(torch.nn.Linear(1, 1)), ParallelConfig())
    initial = plan.build_execution_plan()
    assert plan.rank_for_plan(initial) == 0
    assert plan._local_node_id == "node-b"

    plan.set_membership(("node-a", "node-b", "node-c"))
    expanded = plan.build_execution_plan()
    assert expanded.generation == 1
    assert expanded.previous_generation == 0
    assert plan.rank_for_plan(expanded) == 1
    assert plan.last_reconfiguration.strategies == ("addition",)

    # Re-reading a generation is deterministic and keeps a valid predecessor.
    repeated = plan.build_execution_plan()
    assert repeated.previous_generation == 0


def test_execution_plan_rejects_generation_that_removed_the_local_node():
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=2, microbatch_size=1, max_nodes=2),
        templates=(template(1),),
        node_ids=("node-a", "node-b"),
        rank=0,
    )
    plan.parallelize(torch.nn.Linear(1, 1), ParallelConfig())
    plan.build_execution_plan()
    plan.set_membership(("node-b",))
    target = plan.build_execution_plan()
    with pytest.raises(RecoveryUnavailable, match="local node"):
        plan.rank_for_plan(target)


def test_execution_plan_rejects_runtime_compatibility_mismatch_before_world():
    compatibility = RuntimeCompatibility(
        "oobleck-sha",
        "cornstarch-sha",
        "2.10.0",
        "13.0",
        "2.28",
        "model-digest",
        1,
        1,
        "4.0.0",
        "dataset-digest",
        "gh200-digest",
    )
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=2, microbatch_size=1),
        rank=0,
        compatibility=compatibility,
    )
    plan.parallelize(torch.nn.Linear(1, 1), ParallelConfig())
    execution = plan.build_execution_plan()
    assert execution.compatibility_digest == compatibility.digest
    incompatible = replace(execution, compatibility_digest="different", plan_checksum="")
    with pytest.raises(ValueError, match="compatibility"):
        plan.compile(incompatible)
