from dataclasses import asdict, dataclass, is_dataclass

from simple_parsing import Serializable


@dataclass
class OobleckArguments(Serializable):
    """Data class that contains information about the training job.
    OobleckAgent -> OobleckEngine
    """

    model_name: str
    model_tag: str
    dataset_path: str
    dataset_name: str | None = None
    fault_threshold: int = 3
    microbatch_size: int = 1
    global_microbatch_size: int = 128
    model_args: dict[str, any] | None = None
    steps: int = 50


@dataclass
class OobleckAgentArguments(Serializable):
    """Data class that contains information about the training job.
    OobleckMasterDaemon -> OobleckAgent
    """

    master_ip: str
    master_port: int
    node_ips: list[str]
    job_args: OobleckArguments
    num_workers: int


@dataclass
class DistributedJobConfiguration(Serializable):
    """Data class that contains information about distributed training.
    run.py -> OobleckMasterDaemon
    """

    master_ip: str
    master_port: int
    node_ips: list[str]
    job_args: OobleckArguments
    node_port: int = 22
    username: str | None = None


def flatten_configurations(
    dataclass: OobleckAgentArguments | DistributedJobConfiguration,
) -> dict:
    """Flatten the configuration dataclass into a dict."""
    result = {}
    for k, v in asdict(dataclass).items():
        if is_dataclass(v):
            result.update(flatten_configurations(v))
        else:
            result[k] = v
    return result
