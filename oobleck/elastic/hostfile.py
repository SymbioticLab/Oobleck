"""Optional bootstrap-only SSH host inventory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class InitialHost:
    """Stable bootstrap node identity, SSH address, and fixed local GPU set."""

    node_id: str
    address: str
    gpu_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require complete identity and unique GPU identifiers."""

        if not self.node_id or not self.address or not self.gpu_ids:
            raise ValueError("hostfile node_id, address, and gpu_ids are required")
        if len(self.gpu_ids) != len(set(self.gpu_ids)):
            raise ValueError(f"hostfile node {self.node_id!r} has duplicate GPU IDs")


def load_initial_hostfile(path: str | Path) -> tuple[InitialHost, ...]:
    """Parse ``NODE_ID ADDRESS GPU[,GPU...]`` bootstrap records."""

    target = Path(path)
    hosts = []
    for line_number, raw in enumerate(target.read_text().splitlines(), 1):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        fields = value.split()
        if len(fields) != 3:
            raise ValueError(f"{target}:{line_number}: expected NODE_ID ADDRESS GPU[,GPU...]")
        hosts.append(InitialHost(fields[0], fields[1], tuple(fields[2].split(","))))
    if not hosts:
        raise ValueError(f"initial hostfile {target} contains no nodes")
    node_ids = [item.node_id for item in hosts]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("initial hostfile contains duplicate stable node IDs")
    widths = {len(item.gpu_ids) for item in hosts}
    if len(widths) != 1:
        raise ValueError("initial hostfile nodes must use one fixed per-node TP width")
    return tuple(hosts)


def ssh_agent_command(
    host: InitialHost,
    *,
    agent_script: str | Path,
    worker_script: str | Path,
    master_host: str,
    master_port: int,
    worker_args: Sequence[str] = (),
    remote_python: str = "python",
    socket_directory: str | Path = "/tmp",
    ssh_command: str = "ssh",
) -> tuple[str, ...]:
    """Construct the complete SSH command for one bootstrap agent without side effects.

    The command carries stable node identity, master endpoint, GPU inventory, local-worker socket,
    worker entrypoint, and optional training arguments. Returning argv rather than a shell string
    preserves quoting and lets the launcher supervise processes directly. This is intentionally
    limited to initial bootstrap; membership controls later elastic additions.
    """

    if not 1 <= master_port <= 65535:
        raise ValueError("master_port must be reachable and nonzero for SSH bootstrap")
    socket_path = Path(socket_directory) / f"oobleck-{host.node_id}.sock"
    remote = (
        remote_python,
        str(agent_script),
        "--node-id",
        host.node_id,
        "--master-host",
        master_host,
        "--master-port",
        str(master_port),
        "--gpu-ids",
        *host.gpu_ids,
        "--local-worker-socket",
        str(socket_path),
        "--worker-script",
        str(worker_script),
    )
    if worker_args:
        remote = (*remote, "--worker-args", *worker_args)
    return ssh_command, host.address, *remote


__all__ = ["InitialHost", "load_initial_hostfile", "ssh_agent_command"]
