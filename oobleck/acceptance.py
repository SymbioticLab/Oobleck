"""Machine-readable metrics and verification for dedicated chaos/churn runs."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


_METRIC_LOCK = threading.Lock()


def _metric_int(record: Mapping[str, object], field: str) -> int:
    """Read an integer metric without accepting bool's integer subclass."""

    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"metric {field} must be an integer")
    return value


def _metric_float(record: Mapping[str, object], field: str) -> float:
    """Read a numeric metric and normalize it for aggregation."""

    value = record[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric {field} must be a number")
    return float(value)


def resolve_metrics_path(template: str | Path, *, node_id: str, worker_id: str) -> Path:
    """Expand worker/node/PID placeholders into a collision-resistant output path."""

    value = str(template).format(
        node_id=node_id,
        worker_id=worker_id.replace(":", "-"),
        pid=os.getpid(),
    )
    return Path(value)


def _process_group_count() -> int:
    """Best-effort count live private c10d handles for leak detection."""

    try:
        from torch.distributed import distributed_c10d as c10d

        world = getattr(c10d, "_world", None)
        return 0 if world is None else len(world.pg_map)
    except (ImportError, AttributeError, RuntimeError):
        return 0


def runtime_metric(
    context: Any,
    *,
    event: str,
    node_id: str,
    worker_id: str,
    result: Any = None,
    step_seconds: float = 0.0,
) -> dict[str, object]:
    """Build one machine-readable observation of runtime and recovery health.

    The record binds worker identity to generation, committed transaction progress, plan and
    compatibility checksums, retry count, CUDA allocation, and live process-group count. When
    the current generation has transition metrics, its detection, planning, transfer, activation,
    and balancing measurements are nested into the same record. The function only snapshots
    state; durable JSONL ordering and flushing belong to :func:`append_metric`.
    """

    if not event or step_seconds < 0:
        raise ValueError("metric event and non-negative step duration are required")
    device = getattr(context, "device", torch.device("cpu"))
    cuda_allocated = 0
    cuda_reserved = 0
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        cuda_allocated = int(torch.cuda.memory_allocated(device))
        cuda_reserved = int(torch.cuda.memory_reserved(device))
    reconfiguration = getattr(context.owner_plan, "last_reconfiguration", None)
    strategies = list(getattr(reconfiguration, "strategies", ()))
    history = getattr(context, "recovery_history", ())
    transition = history[-1] if history else None
    metric: dict[str, object] = {
        "schema_version": 1,
        "event": event,
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "node_id": node_id,
        "worker_id": worker_id,
        "generation": int(context.generation),
        "committed_step": int(context.committed_step),
        "attempts": 0 if result is None else int(result.attempts),
        "step_seconds": step_seconds,
        "strategies": strategies,
        "plan_checksum": str(context.execution_plan.plan_checksum),
        "compatibility_digest": str(context.execution_plan.compatibility_digest or ""),
        "cuda_allocated_bytes": cuda_allocated,
        "cuda_reserved_bytes": cuda_reserved,
        "process_group_count": _process_group_count(),
        "distributed_initialized": bool(
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ),
    }
    if transition is not None and transition.generation == context.generation:
        metric["recovery"] = {
            "detection_seconds": transition.detection_seconds,
            "configuration_planning_seconds": transition.configuration_planning_seconds,
            "teardown_seconds": transition.teardown_seconds,
            "state_planning_seconds": transition.state_planning_seconds,
            "state_transfer_seconds": transition.state_transfer_seconds,
            "straggler_round_seconds": transition.straggler_round_seconds,
            "world_initialization_seconds": transition.world_initialization_seconds,
            "activation_seconds": transition.activation_seconds,
            "total_seconds": transition.total_seconds,
            "source_scheduling_error_bytes": transition.source_scheduling_error_bytes,
            "removed_members": [
                {"agent_id": agent_id, "incarnation_id": incarnation_id}
                for agent_id, incarnation_id in transition.removed_members
            ],
            "added_members": [
                {"agent_id": agent_id, "incarnation_id": incarnation_id}
                for agent_id, incarnation_id in transition.added_members
            ],
            "graceful_cutover": transition.graceful_cutover,
            "cutover_committed_step": transition.cutover_committed_step,
        }
    return metric


def append_metric(path: str | Path, metric: Mapping[str, object]) -> None:
    """Append and flush one canonical JSONL record under a process-local lock."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(metric), sort_keys=True, separators=(",", ":")) + "\n"
    with _METRIC_LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()


def load_metrics(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    """Load JSONL records from every worker output with source-aware validation."""

    records: list[dict[str, object]] = []
    for path in paths:
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number} metric must be a JSON object")
                records.append(value)
    return records


def verify_churn_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    required_strategies: Sequence[str] = (),
    require_replay: bool = True,
    require_clean_close: bool = False,
    max_cuda_reserved_growth_bytes: int = 256 * 1024 * 1024,
    max_process_group_growth: int = 16,
) -> dict[str, object]:
    """Validate a multi-worker churn run against transactional and resource invariants.

    Records are grouped by stable worker identity and ordered by timestamp. Each worker must
    observe nondecreasing generations and strictly consecutive committed steps, while retries
    prove interrupted-batch replay when required. Recovery-strategy coverage, CUDA-reserved
    growth, process-group growth, step latency, and optional clean shutdown are checked against
    caller bounds. The returned summary is suitable for CI artifacts and benchmark comparison.
    """

    if not records:
        raise ValueError("at least one worker metric is required")
    if max_cuda_reserved_growth_bytes < 0 or max_process_group_growth < 0:
        raise ValueError("memory and process-group growth bounds must be non-negative")
    by_worker: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        required = {
            "schema_version",
            "event",
            "timestamp_ns",
            "worker_id",
            "generation",
            "committed_step",
            "attempts",
            "step_seconds",
            "strategies",
            "plan_checksum",
            "compatibility_digest",
            "cuda_reserved_bytes",
            "process_group_count",
            "distributed_initialized",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"metric is missing fields {sorted(missing)}")
        if record["schema_version"] != 1:
            raise ValueError("unsupported worker metric schema")
        by_worker.setdefault(str(record["worker_id"]), []).append(record)

    observed_strategies: set[str] = set()
    replay_count = 0
    graceful_transition_count = 0
    maximum_growth = 0
    maximum_group_growth = 0
    worker_summaries: dict[str, object] = {}
    for worker_id, worker_records in sorted(by_worker.items()):
        ordered = sorted(worker_records, key=lambda item: _metric_int(item, "timestamp_ns"))
        generations = [_metric_int(item, "generation") for item in ordered]
        if generations != sorted(generations):
            raise AssertionError(f"worker {worker_id} generations moved backwards")
        steps = [item for item in ordered if item["event"] == "step"]
        committed = [_metric_int(item, "committed_step") for item in steps]
        step_durations = [_metric_float(item, "step_seconds") for item in steps]
        if any(duration < 0 for duration in step_durations):
            raise ValueError("metric step_seconds must be non-negative")
        if any(right != left + 1 for left, right in zip(committed, committed[1:])):
            raise AssertionError(f"worker {worker_id} committed steps skipped or duplicated")
        replay_count += sum(_metric_int(item, "attempts") > 1 for item in steps)
        graceful: dict[int, int] = {}
        for item in ordered:
            recovery = item.get("recovery")
            if not isinstance(recovery, Mapping) or not recovery.get("graceful_cutover"):
                continue
            removed_members = recovery.get("removed_members")
            added_members = recovery.get("added_members")
            if removed_members != [] or not isinstance(added_members, list) or not added_members:
                raise AssertionError("graceful cutover must contain additions and no removals")
            for field, members in (
                ("removed_members", removed_members),
                ("added_members", added_members),
            ):
                if not isinstance(members, list) or not all(
                    isinstance(member, Mapping)
                    and set(member) == {"agent_id", "incarnation_id"}
                    and all(isinstance(value, str) and value for value in member.values())
                    for member in members
                ):
                    raise ValueError(f"recovery {field} identities are invalid")
            cutover = recovery.get("cutover_committed_step")
            if not isinstance(cutover, int) or isinstance(cutover, bool) or cutover < 0:
                raise ValueError("graceful cutover committed step must be non-negative")
            graceful[_metric_int(item, "generation")] = cutover
        for generation, cutover in graceful.items():
            resumed = [
                item
                for item in steps
                if _metric_int(item, "generation") == generation
                and _metric_int(item, "committed_step") > cutover
            ]
            if not resumed:
                raise AssertionError("graceful addition did not resume after the cutover step")
            first_resumed = resumed[0]
            if (
                _metric_int(first_resumed, "committed_step") != cutover + 1
                or _metric_int(first_resumed, "attempts") != 1
            ):
                raise AssertionError("graceful addition replayed or skipped the next logical batch")
        graceful_transition_count += len(graceful)
        for item in ordered:
            strategies = item["strategies"]
            if not isinstance(strategies, list) or not all(
                isinstance(strategy, str) for strategy in strategies
            ):
                raise ValueError("metric strategies must be a list of strings")
            observed_strategies.update(strategies)
        reserved = [_metric_int(item, "cuda_reserved_bytes") for item in ordered]
        live_reserved = [
            value for item, value in zip(ordered, reserved) if item["event"] != "closed"
        ]
        growth = max(0, max(live_reserved) - live_reserved[0]) if live_reserved else 0
        maximum_growth = max(maximum_growth, growth)
        if growth > max_cuda_reserved_growth_bytes:
            raise AssertionError(f"worker {worker_id} CUDA reserved memory grew by {growth} bytes")
        group_counts = [_metric_int(item, "process_group_count") for item in ordered]
        live_group_counts = [
            count for item, count in zip(ordered, group_counts) if item["event"] != "closed"
        ]
        group_growth = max(live_group_counts) - live_group_counts[0] if live_group_counts else 0
        maximum_group_growth = max(maximum_group_growth, group_growth)
        if group_growth > max_process_group_growth:
            raise AssertionError(f"worker {worker_id} process-group count grew by {group_growth}")
        closed = [item for item in ordered if item["event"] == "closed"]
        if require_clean_close and (
            not closed
            or _metric_int(closed[-1], "process_group_count") != 0
            or bool(closed[-1]["distributed_initialized"])
        ):
            raise AssertionError(f"worker {worker_id} did not retire all process groups")
        worker_summaries[worker_id] = {
            "records": len(ordered),
            "steps": len(steps),
            "first_generation": generations[0],
            "last_generation": generations[-1],
            "first_committed_step": committed[0] if committed else 0,
            "last_committed_step": committed[-1] if committed else 0,
            "average_step_seconds": (
                sum(step_durations) / len(step_durations) if step_durations else 0.0
            ),
            "cuda_reserved_growth_bytes": growth,
            "process_group_growth": group_growth,
        }

    missing_strategies = set(required_strategies) - observed_strategies
    if missing_strategies:
        raise AssertionError(f"recovery strategies were not observed: {sorted(missing_strategies)}")
    if require_replay and replay_count == 0:
        raise AssertionError("no interrupted logical batch was replayed")
    return {
        "schema_version": 1,
        "workers": worker_summaries,
        "observed_strategies": sorted(observed_strategies),
        "replayed_steps": replay_count,
        "graceful_transitions": graceful_transition_count,
        "maximum_cuda_reserved_growth_bytes": maximum_growth,
        "maximum_process_group_growth": maximum_group_growth,
    }


__all__ = [
    "append_metric",
    "load_metrics",
    "resolve_metrics_path",
    "runtime_metric",
    "verify_churn_metrics",
]
