"""Serializable Tyro command configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MasterServiceConfig:
    host: str = "127.0.0.1"
    port: int = 0
    heartbeat_interval_s: float = 1.0
    lease_timeout_s: float = 5.0
    max_frame_bytes: int = 1 << 20
    max_nodes: int = 16

    def __post_init__(self) -> None:
        if not self.host or not 0 <= self.port <= 65535:
            raise ValueError("master host/port is invalid")
        if self.heartbeat_interval_s <= 0 or self.lease_timeout_s <= self.heartbeat_interval_s:
            raise ValueError("lease timeout must exceed the positive heartbeat interval")
        if self.max_frame_bytes < 256 or self.max_nodes < 1:
            raise ValueError("frame limit and max_nodes must be positive")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    node_id: str
    master_host: str = "127.0.0.1"
    master_port: int = 0
    gpu_ids: tuple[str, ...] = ("0",)
    local_worker_socket: Path | None = None
    worker_script: Path | None = None
    worker_args: tuple[str, ...] = ()
    heartbeat_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.node_id or not self.master_host or not 1 <= self.master_port <= 65535:
            raise ValueError("agent identity and master address are required")
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must be non-empty and unique")
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        if self.worker_script is not None and self.local_worker_socket is None:
            raise ValueError("worker_script requires local_worker_socket")


@dataclass(frozen=True, slots=True)
class TrainingLaunchConfig:
    training_script: Path
    script_args: tuple[str, ...] = ()
    max_nodes: int = 1
    tensor_parallel_size: int = 1
    initial_hostfile: Path | None = None
    agent_script: Path | None = None
    master_host: str = "127.0.0.1"
    master_port: int = 0
    remote_python: str = "python"
    ssh_command: str = "ssh"
    socket_directory: Path = Path("/tmp")

    def __post_init__(self) -> None:
        if self.max_nodes < 1 or self.tensor_parallel_size < 1:
            raise ValueError("max_nodes and tensor_parallel_size must be positive")
        if self.initial_hostfile is not None and (
            self.agent_script is None or not self.master_host or not 1 <= self.master_port <= 65535
        ):
            raise ValueError("SSH bootstrap requires agent_script and a reachable master host/port")


@dataclass(frozen=True, slots=True)
class ProfileCommandConfig:
    output: Path
    model: str
    profile: Path | None = None
    measurement_factory: str | None = None
    profile_output: Path | None = None
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    microbatch_size: int = 1
    resource_counts: tuple[int, ...] = (1,)
    hardware: str | None = None
    cornstarch_version: str | None = None
    warmup_steps: int = 2
    measurement_steps: int = 5
    device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.model or self.tensor_parallel_size < 1 or self.microbatch_size < 1:
            raise ValueError("profile model and parallel sizes are required")
        if not self.resource_counts or any(item < 1 for item in self.resource_counts):
            raise ValueError("resource_counts must contain positive values")
        if self.profile is not None and self.measurement_factory is not None:
            raise ValueError("provide either profile or measurement_factory, not both")
        if self.warmup_steps < 0 or self.measurement_steps < 1:
            raise ValueError("profiling step counts are invalid")
        if self.device_memory_bytes is not None and self.device_memory_bytes < 1:
            raise ValueError("device_memory_bytes must be positive when supplied")


@dataclass(frozen=True, slots=True)
class DrainConfig:
    node_id: str
    master_host: str = "127.0.0.1"
    master_port: int = 0

    def __post_init__(self) -> None:
        if not self.node_id or not self.master_host or not 1 <= self.master_port <= 65535:
            raise ValueError("drain requires a node identity and master address")


@dataclass(frozen=True, slots=True)
class InspectMembershipConfig:
    master_host: str = "127.0.0.1"
    master_port: int = 0

    def __post_init__(self) -> None:
        if not self.master_host or not 1 <= self.master_port <= 65535:
            raise ValueError("membership inspection requires a master address")


@dataclass(frozen=True, slots=True)
class ChaosConfig:
    target_node_id: str
    pid: int = 0
    confirmation: str = ""

    def __post_init__(self) -> None:
        if self.pid < 1 or self.confirmation != f"FAIL:{self.target_node_id}":
            raise ValueError("pid must be positive and confirmation must be FAIL:<node-id>")
