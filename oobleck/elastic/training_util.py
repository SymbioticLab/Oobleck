from dataclasses import dataclass, fields, is_dataclass

from simple_parsing import Serializable


@dataclass
class DistributedArguments(Serializable):
    master_ip: str
    master_port: int
    node_ips: list[str]
    node_port: int = 22
    num_workers: int = 0
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
    def __init__(
        self,
        model_name: str,
        model_tag: str,
        dataset_path: str,
        dataset_name: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_tag = model_tag
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        for key, value in kwargs.items():
            setattr(self, key, value)

    model_name: str
    model_tag: str
    dataset_path: str
    dataset_name: str | None = None
    # Other arbitrary model configuration can be here..


@dataclass
class OobleckArguments(Serializable):
    dist: DistributedArguments
    job: JobArguments
    model: ModelArguments
