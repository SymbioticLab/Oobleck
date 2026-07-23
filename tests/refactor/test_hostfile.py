from __future__ import annotations

import pytest

from oobleck.config import TrainingLaunchConfig
from oobleck.elastic import load_initial_hostfile, ssh_agent_command


def test_initial_hostfile_is_strict_and_builds_agent_commands(tmp_path):
    path = tmp_path / "hosts.txt"
    path.write_text("# stable node inventory\nnode-a 10.0.0.1 0,1\nnode-b 10.0.0.2 0,1\n")
    hosts = load_initial_hostfile(path)
    assert [item.node_id for item in hosts] == ["node-a", "node-b"]
    command = ssh_agent_command(
        hosts[0],
        agent_script="run_agent.py",
        worker_script="pretrain.py",
        master_host="10.0.0.9",
        master_port=29600,
        worker_args=("--split", "train"),
    )
    assert command[:3] == ("ssh", "10.0.0.1", "python")
    assert "--node-id" in command and "node-a" in command
    assert command[-3:] == ("--worker-args", "--split", "train")


def test_initial_hostfile_rejects_duplicate_ids_and_mixed_tp_width(tmp_path):
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("node-a host-a 0\nnode-a host-b 0\n")
    with pytest.raises(ValueError, match="duplicate stable node"):
        load_initial_hostfile(duplicate)

    mixed = tmp_path / "mixed.txt"
    mixed.write_text("node-a host-a 0\nnode-b host-b 0,1\n")
    with pytest.raises(ValueError, match="fixed per-node TP width"):
        load_initial_hostfile(mixed)


def test_ssh_launch_configuration_requires_master_and_agent_script(tmp_path):
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("node-a host-a 0\n")
    with pytest.raises(ValueError, match="SSH bootstrap"):
        TrainingLaunchConfig(
            training_script=tmp_path / "train.py",
            initial_hostfile=hostfile,
        )
