from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class OobleckArguments(Serializable):
    model_name: str
    model_tag: str
    dataset_path: str
    dataset_name: str | None = None
    fault_threshold: int = 3
    microbatch_size: int = 1
    global_microbatch_size: int = 128
    model_args: dict[str, any] | None = None
