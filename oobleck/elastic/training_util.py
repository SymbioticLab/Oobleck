from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class TrainingArguments(Serializable):
    model_name: str
    model_tag: str
    dataset_path: str
    dataset_name: str | None = None
    fault_threshold: int = 3
    model_args: dict[str, any] | None = None
    hf_training_args: dict[str, any] | None = None
