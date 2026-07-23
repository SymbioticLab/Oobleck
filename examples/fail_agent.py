"""Opt-in demo failure helper that never targets arbitrary processes."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureConfig:
    pid: int
    expected_node_id: str
    confirmation: str


def main() -> None:
    import tyro

    config = tyro.cli(FailureConfig)
    if config.confirmation != f"FAIL:{config.expected_node_id}":
        raise SystemExit("confirmation must be FAIL:<expected-node-id>")
    cmdline = open(f"/proc/{config.pid}/cmdline", "rb").read().decode(errors="replace")
    if "examples/run_agent.py" not in cmdline or config.expected_node_id not in cmdline:
        raise SystemExit("target is not the expected disposable Oobleck example agent")
    os.kill(config.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
