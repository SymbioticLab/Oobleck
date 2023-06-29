from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class TrainingArguments(Serializable):
    model_name: str
    dataset_path: str
    microbatch_size: int
    global_gradient_accumulation_steps: int
    dataset_name: str | None = None
    fault_threshold: int = 3
