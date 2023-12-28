from dataclasses import dataclass


@dataclass
class DistArgs:
    # IP addresses of all agents. In command line, IPs are separated by space.
    agent_ips: list[str]
    # Total number of ranks.
    world_size: int
    # torch.distributed backend.
    backend: str = "nccl"
    # Number of ranks for tensor parallelism.
    tensor_parallel_size: int = 1


@dataclass
class TrainingArgs:
    model: str
    dataset_name: str
    mixed_precision: str
    microbatch_size: int
    global_batch_size: int
    dataset_path: str | None = None
