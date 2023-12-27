from dataclasses import dataclass


@dataclass
class DistArgs:
    world_size: int
    agent_ips: list[str]
    local_rank: int
    backend: str = "nccl"
    tensor_parallel_size: int = 1


@dataclass
class TrainingArgs:
    model: str
    dataset_name: str
    mixed_precision: str
    microbatch_size: int
    global_batch_size: int
    dataset_path: str | None = None
