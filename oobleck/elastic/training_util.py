from dataclasses import dataclass, field
from typing import Any

from simple_parsing import Serializable


@dataclass
class DistributedArguments(Serializable):
    master_ip: str
    master_port: int
    node_ips: list[str]
    node_port: int = 22
    num_workers: int = 1
    num_agents_per_node: int = 1
    username: str | None = None


@dataclass
class JobArguments(Serializable):
    fault_threshold: int = 3
    microbatch_size: int = 1
    global_microbatch_size: int = 128
    steps: int = 50


@dataclass
class ModelArguments(Serializable):
    model_name: str
    model_tag: str
    dataset_path: str
    dataset_name: str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class OobleckArguments(Serializable):
    dist: DistributedArguments
    job: JobArguments
    model: ModelArguments
