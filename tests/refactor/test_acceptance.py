from __future__ import annotations

import asyncio

import pytest

from benchmarks.chaos_acceptance import ChaosAcceptanceConfig, ChaosEvent, run

from oobleck.acceptance import append_metric, load_metrics, verify_churn_metrics
from oobleck.elastic import ControlStatus, MembershipSnapshot, NodeIdentity


def metric(
    timestamp: int,
    *,
    event: str = "step",
    generation: int = 1,
    step: int = 1,
    attempts: int = 1,
    strategies: list[str] | None = None,
    reserved: int = 100,
    groups: int = 1,
    initialized: bool = True,
):
    return {
        "schema_version": 1,
        "event": event,
        "timestamp_ns": timestamp,
        "worker_id": "node-a:gpu-0",
        "generation": generation,
        "committed_step": step,
        "attempts": attempts,
        "step_seconds": 0.1,
        "strategies": strategies or [],
        "plan_checksum": f"plan-{generation}",
        "compatibility_digest": "runtime",
        "cuda_reserved_bytes": reserved,
        "process_group_count": groups,
        "distributed_initialized": initialized,
    }


def test_churn_verifier_proves_replay_strategies_memory_and_clean_close(tmp_path):
    records = [
        metric(1, step=1),
        metric(2, generation=2, step=2, attempts=2, strategies=["simple"]),
        metric(3, generation=3, step=3, strategies=["borrow", "merge"], reserved=120),
        metric(
            4,
            event="closed",
            generation=3,
            step=3,
            reserved=120,
            groups=0,
            initialized=False,
        ),
    ]
    path = tmp_path / "worker.jsonl"
    for record in records:
        append_metric(path, record)
    observed = load_metrics([path])
    result = verify_churn_metrics(
        observed,
        required_strategies=("simple", "borrow", "merge"),
        require_clean_close=True,
        max_cuda_reserved_growth_bytes=20,
    )
    assert result["replayed_steps"] == 1
    assert result["observed_strategies"] == ["borrow", "merge", "simple"]
    assert result["maximum_cuda_reserved_growth_bytes"] == 20


def test_churn_verifier_rejects_commit_gaps_memory_growth_and_group_leaks():
    with pytest.raises(AssertionError, match="skipped or duplicated"):
        verify_churn_metrics(
            [metric(1, step=1), metric(2, step=3, attempts=2)],
            require_replay=False,
        )
    with pytest.raises(AssertionError, match="CUDA reserved memory"):
        verify_churn_metrics(
            [metric(1, step=1), metric(2, step=2, attempts=2, reserved=200)],
            max_cuda_reserved_growth_bytes=10,
        )
    with pytest.raises(AssertionError, match="process-group count"):
        verify_churn_metrics(
            [metric(1, step=1, groups=1), metric(2, step=2, groups=4)],
            require_replay=False,
            max_process_group_growth=2,
        )
    with pytest.raises(AssertionError, match="retire all process groups"):
        verify_churn_metrics(
            [metric(1, step=1, attempts=2), metric(2, event="closed", groups=1)],
            require_clean_close=True,
        )


def test_churn_verifier_handles_twenty_five_generation_campaign():
    records = [
        metric(
            generation,
            generation=generation,
            step=generation,
            attempts=2 if generation == 2 else 1,
        )
        for generation in range(1, 26)
    ]
    result = verify_churn_metrics(records)
    assert result["workers"]["node-a:gpu-0"]["records"] == 25
    assert result["workers"]["node-a:gpu-0"]["last_generation"] == 25


def test_chaos_runner_injects_cascade_before_activation_and_verifies_metrics(tmp_path):
    def status(
        generation: int,
        nodes: tuple[str, ...],
        active: bool,
        active_generation: int | None = None,
    ) -> ControlStatus:
        snapshot = MembershipSnapshot(
            generation,
            tuple(
                NodeIdentity(node, f"{node}-{generation}", ("127.0.0.1",), ("0",)) for node in nodes
            ),
            (),
        )
        agents = nodes if active else ()
        if active_generation is None:
            active_generation = generation if active else generation - 1
        return ControlStatus(snapshot, active_generation, agents, agents)

    statuses = [
        status(1, ("a", "b", "c", "d"), True),
        status(2, ("a", "b", "c"), False),
        status(3, ("a", "b"), False, 1),
        status(3, ("a", "b"), True),
    ]

    async def read_status(host: str, port: int) -> ControlStatus:
        return statuses.pop(0)

    commands = []

    async def run_commands(argv):
        commands.append(tuple(tuple(item) for item in argv))

    metrics_path = tmp_path / "metrics.jsonl"
    for record in (
        metric(1, step=1),
        metric(2, generation=3, step=2, attempts=2, strategies=["simple", "borrow", "merge"]),
    ):
        append_metric(metrics_path, record)
    config = ChaosAcceptanceConfig(
        "master",
        29600,
        ("a", "b", "c", "d"),
        (
            ChaosEvent(
                "cascading-failure",
                (("kill", "d"),),
                ("a", "b", "c"),
                (("kill", "c"),),
                ("a", "b"),
                1.0,
            ),
        ),
        (metrics_path,),
        poll_interval_s=0.001,
    )
    result = asyncio.run(run(config, command_runner=run_commands, status_reader=read_status))
    assert commands == [(("kill", "d"),), (("kill", "c"),)]
    assert result["initial_generation"] == 1
    assert result["final_generation"] == 3
    assert result["events"][0]["cascaded"] is True
    assert result["verification"]["replayed_steps"] == 1
