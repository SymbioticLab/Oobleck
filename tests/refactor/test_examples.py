from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from oobleck.config import AgentConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_agent_worker_configuration_requires_local_ipc(tmp_path):
    script = REPOSITORY_ROOT / "examples" / "pretrain_llm.py"
    try:
        AgentConfig(
            node_id="node-a",
            master_port=29600,
            worker_script=script,
        )
    except ValueError as exc:
        assert "local_worker_socket" in str(exc)
    else:
        raise AssertionError("worker launch without local IPC should be rejected")

    config = AgentConfig(
        node_id="node-a",
        master_port=29600,
        local_worker_socket=tmp_path / "agent.sock",
        worker_script=script,
        addresses=("127.0.0.1",),
    )
    assert config.worker_script == script
    assert config.addresses == ("127.0.0.1",)


def test_one_node_master_agent_worker_subprocess_smoke():
    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "examples" / "run_local.py")],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "committed_step=1 generation=1 attempts=1" in result.stdout
    assert "local deployment completed; final_generation=2" in result.stdout
