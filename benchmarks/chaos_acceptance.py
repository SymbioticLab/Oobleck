"""Optional multi-node NCCL chaos and long-churn validation orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

import torch

from oobleck.acceptance import load_metrics, verify_churn_metrics
from oobleck.elastic import ControlStatus, inspect_status


@dataclass(frozen=True, slots=True)
class ChaosEvent:
    name: str
    commands: tuple[tuple[str, ...], ...]
    expected_nodes: tuple[str, ...]
    cascade_commands: tuple[tuple[str, ...], ...] = ()
    cascade_expected_nodes: tuple[str, ...] = ()
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.name or not self.commands or not self.expected_nodes:
            raise ValueError("chaos event requires a name, commands, and expected nodes")
        if any(not command or any(not item for item in command) for command in self.commands):
            raise ValueError("chaos commands must be non-empty argv arrays")
        if bool(self.cascade_commands) != bool(self.cascade_expected_nodes):
            raise ValueError("cascade commands and expected nodes must be supplied together")
        if self.timeout_s <= 0:
            raise ValueError("chaos event timeout must be positive")


@dataclass(frozen=True, slots=True)
class ChaosAcceptanceConfig:
    master_host: str
    master_port: int
    initial_nodes: tuple[str, ...]
    events: tuple[ChaosEvent, ...]
    metric_files: tuple[Path, ...]
    required_strategies: tuple[str, ...] = ("simple", "borrow", "merge")
    poll_interval_s: float = 0.1
    max_cuda_reserved_growth_bytes: int = 256 * 1024 * 1024
    max_process_group_growth: int = 16
    require_clean_close: bool = False

    def __post_init__(self) -> None:
        if not self.master_host or not 1 <= self.master_port <= 65535:
            raise ValueError("a reachable master host and port are required")
        if not self.initial_nodes or not self.events or not self.metric_files:
            raise ValueError("initial nodes, events, and metric files are required")
        if (
            self.poll_interval_s <= 0
            or self.max_cuda_reserved_growth_bytes < 0
            or self.max_process_group_growth < 0
        ):
            raise ValueError("poll interval and memory growth bound are invalid")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ChaosAcceptanceConfig":
        expected = {
            "schema_version",
            "master_host",
            "master_port",
            "initial_nodes",
            "events",
            "metric_files",
            "required_strategies",
            "poll_interval_s",
            "max_cuda_reserved_growth_bytes",
            "require_clean_close",
            "max_process_group_growth",
        }
        if set(value) != expected or value["schema_version"] != 1:
            raise ValueError("chaos manifest fields or schema version are invalid")

        def commands(raw: object) -> tuple[tuple[str, ...], ...]:
            if not isinstance(raw, list) or not all(
                isinstance(item, list)
                and item
                and all(isinstance(part, str) and part for part in item)
                for item in raw
            ):
                raise ValueError("commands must be a list of non-empty string argv arrays")
            return tuple(tuple(str(part) for part in item) for item in raw)

        def sequence(raw: object, name: str) -> list[object]:
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list")
            return raw

        def strings(raw: object, name: str) -> tuple[str, ...]:
            values = sequence(raw, name)
            if not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
            return tuple(str(item) for item in values)

        def string(raw: object, name: str) -> str:
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"{name} must be a non-empty string")
            return raw

        def integer(raw: object, name: str) -> int:
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise ValueError(f"{name} must be an integer")
            return raw

        def number(raw: object, name: str) -> float:
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise ValueError(f"{name} must be a number")
            return float(raw)

        def boolean(raw: object, name: str) -> bool:
            if not isinstance(raw, bool):
                raise ValueError(f"{name} must be a boolean")
            return raw

        raw_events = value["events"]
        if not isinstance(raw_events, list):
            raise ValueError("events must be a list")
        events = []
        for raw in raw_events:
            if not isinstance(raw, dict) or set(raw) != {
                "name",
                "commands",
                "expected_nodes",
                "cascade_commands",
                "cascade_expected_nodes",
                "timeout_s",
            }:
                raise ValueError("chaos event fields are invalid")
            events.append(
                ChaosEvent(
                    string(raw["name"], "name"),
                    commands(raw["commands"]),
                    tuple(str(item) for item in sequence(raw["expected_nodes"], "expected_nodes")),
                    commands(raw["cascade_commands"]),
                    strings(raw["cascade_expected_nodes"], "cascade_expected_nodes"),
                    number(raw["timeout_s"], "timeout_s"),
                )
            )
        return cls(
            master_host=string(value["master_host"], "master_host"),
            master_port=integer(value["master_port"], "master_port"),
            initial_nodes=strings(value["initial_nodes"], "initial_nodes"),
            events=tuple(events),
            metric_files=tuple(
                Path(str(item)) for item in sequence(value["metric_files"], "metric_files")
            ),
            required_strategies=strings(value["required_strategies"], "required_strategies"),
            poll_interval_s=number(value["poll_interval_s"], "poll_interval_s"),
            max_cuda_reserved_growth_bytes=integer(
                value["max_cuda_reserved_growth_bytes"], "max_cuda_reserved_growth_bytes"
            ),
            max_process_group_growth=integer(
                value["max_process_group_growth"], "max_process_group_growth"
            ),
            require_clean_close=boolean(value["require_clean_close"], "require_clean_close"),
        )


def load_config(path: str | Path) -> ChaosAcceptanceConfig:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("chaos manifest must be a JSON object")
    return ChaosAcceptanceConfig.from_mapping(value)


async def _run_commands(commands: Sequence[Sequence[str]]) -> None:
    processes = [
        await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for command in commands
    ]
    outputs = await asyncio.gather(*(process.communicate() for process in processes))
    failures = []
    for command, process, (stdout, stderr) in zip(commands, processes, outputs):
        if process.returncode:
            failures.append(
                {
                    "command": list(command),
                    "returncode": process.returncode,
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                }
            )
    if failures:
        raise RuntimeError(f"chaos commands failed: {failures}")


async def _wait_for_status(
    config: ChaosAcceptanceConfig,
    predicate: Callable[[ControlStatus], bool],
    *,
    timeout_s: float,
    status_reader: Callable[[str, int], Awaitable[ControlStatus]],
) -> ControlStatus:
    deadline = time.monotonic() + timeout_s
    last: ControlStatus | None = None
    while time.monotonic() < deadline:
        last = await status_reader(config.master_host, config.master_port)
        if predicate(last):
            return last
        await asyncio.sleep(config.poll_interval_s)
    raise TimeoutError(f"control status did not converge; last={last}")


def _cornstarch_identity() -> tuple[str, str]:
    try:
        distribution = importlib.metadata.distribution("cornstarch")
        version = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        revision = str(direct_url.get("vcs_info", {}).get("commit_id", ""))
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
        revision = ""
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    for dependency in project.read_text().splitlines():
        marker = "Cornstarch.git@"
        if marker in dependency:
            declared = dependency.split(marker, 1)[1][:40]
            if revision and revision != declared:
                raise RuntimeError(
                    f"installed Cornstarch {revision} does not match declared pin {declared}"
                )
            revision = declared
            break
    return version, revision or "unknown"


def _environment(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    cornstarch_version, cornstarch_revision = _cornstarch_identity()
    try:
        datasets_version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError:
        datasets_version = "unavailable"
    return {
        "oobleck_commit": commit,
        "cornstarch_version": cornstarch_version,
        "cornstarch_revision": cornstarch_revision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": (
            list(torch.cuda.nccl.version())  # pyright: ignore[reportAttributeAccessIssue]
            if torch.cuda.is_available() and torch.distributed.is_nccl_available()
            else None
        ),
        "datasets_version": datasets_version,
        "hardware": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "compatibility_digests": sorted(
            {str(record["compatibility_digest"]) for record in records}
        ),
        "plan_checksums": sorted({str(record["plan_checksum"]) for record in records}),
    }


async def run(
    config: ChaosAcceptanceConfig,
    *,
    command_runner: Callable[[Sequence[Sequence[str]]], Awaitable[None]] = _run_commands,
    status_reader: Callable[[str, int], Awaitable[ControlStatus]] = inspect_status,
) -> dict[str, object]:
    initial = await _wait_for_status(
        config,
        lambda status: (
            status.active
            and tuple(sorted(node.agent_id for node in status.snapshot.nodes))
            == tuple(sorted(config.initial_nodes))
        ),
        timeout_s=max(event.timeout_s for event in config.events),
        status_reader=status_reader,
    )
    timeline = []
    current_generation = initial.generation
    for event in config.events:
        started = time.monotonic()
        await command_runner(event.commands)
        proposal = await _wait_for_status(
            config,
            lambda status: (
                status.generation > current_generation
                and tuple(sorted(node.agent_id for node in status.snapshot.nodes))
                == tuple(sorted(event.expected_nodes))
            ),
            timeout_s=event.timeout_s,
            status_reader=status_reader,
        )
        detected_seconds = time.monotonic() - started
        target_nodes = event.expected_nodes
        if event.cascade_commands:
            if proposal.active:
                raise AssertionError(
                    f"event {event.name} activated before cascading failure was injected"
                )
            await command_runner(event.cascade_commands)
            previous_generation = proposal.generation
            proposal = await _wait_for_status(
                config,
                lambda status: (
                    status.generation > previous_generation
                    and tuple(sorted(node.agent_id for node in status.snapshot.nodes))
                    == tuple(sorted(event.cascade_expected_nodes))
                ),
                timeout_s=event.timeout_s,
                status_reader=status_reader,
            )
            if proposal.active_generation >= previous_generation:
                raise AssertionError(
                    f"event {event.name} activated its intermediate generation before cascade"
                )
            target_nodes = event.cascade_expected_nodes
        active = await _wait_for_status(
            config,
            lambda status: (
                status.generation == proposal.generation
                and status.active
                and tuple(sorted(node.agent_id for node in status.snapshot.nodes))
                == tuple(sorted(target_nodes))
            ),
            timeout_s=event.timeout_s,
            status_reader=status_reader,
        )
        timeline.append(
            {
                "name": event.name,
                "generation": active.generation,
                "members": list(target_nodes),
                "detection_seconds": detected_seconds,
                "recovery_seconds": time.monotonic() - started,
                "cascaded": bool(event.cascade_commands),
            }
        )
        current_generation = active.generation

    records = load_metrics(config.metric_files)
    verification = verify_churn_metrics(
        records,
        required_strategies=config.required_strategies,
        require_replay=True,
        require_clean_close=config.require_clean_close,
        max_cuda_reserved_growth_bytes=config.max_cuda_reserved_growth_bytes,
        max_process_group_growth=config.max_process_group_growth,
    )
    return {
        "schema_version": 1,
        "scenario": "multi-node-nccl-chaos-churn",
        "initial_generation": initial.generation,
        "final_generation": current_generation,
        "events": timeline,
        "verification": verification,
        "environment": _environment(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the manifest's explicit failure/drain/restart argv arrays",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.manifest)
    if not arguments.execute:
        raise SystemExit("refusing to execute chaos commands without --execute")
    result = asyncio.run(run(config))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
