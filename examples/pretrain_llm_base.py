"""Offline-friendly training setup using Oobleck's public DataLoader path."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset

from oobleck import OobleckConfig, OobleckParallelizationPlan


class FakeTextDataset(Dataset):
    def __init__(
        self, samples: int = 128, sequence_length: int = 16, vocab_size: int = 256
    ) -> None:
        generator = torch.Generator().manual_seed(7)
        self.tokens = torch.randint(vocab_size, (samples, sequence_length), generator=generator)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.tokens[index]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


@dataclass
class ExampleParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: None = None
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1


def build_dataset(
    name: str | None = None,
    configuration: str | None = None,
    split: str = "train",
    *,
    vocab_size: int = 256,
):
    if name is None:
        return FakeTextDataset(vocab_size=vocab_size)
    from datasets import load_dataset

    return load_dataset(name, configuration, split=split)


def _build_model(model_backend: str) -> tuple[torch.nn.Module, object, object | None, int]:
    if model_backend == "tiny":
        return TinyLanguageModel(), ExampleParallelConfig(), None, 256
    if model_backend != "cornstarch":
        raise ValueError("model_backend must be 'tiny' or 'cornstarch'")
    try:
        from cornstarch.distributed import ParallelConfig, ParallelizationPlan
        from cornstarch.models import from_hf_config
        from transformers import LlamaConfig
    except ImportError as exc:
        raise RuntimeError(
            "the Cornstarch model backend requires the pinned refactor-oobleck dependency"
        ) from exc

    hf_config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = from_hf_config(hf_config, model_kind="language")
    model.set_random_init()
    cornstarch_plan = ParallelizationPlan(global_ranks=[0])
    return model, ParallelConfig(tensor_parallel_size=1), cornstarch_plan, 64


def build_training(
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    device: str = "cuda",
    model_backend: str = "tiny",
):
    model, parallel_config, cornstarch_plan, vocab_size = _build_model(model_backend)
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=8, microbatch_size=2, max_nodes=1, seed=42),
        cornstarch_plan=cornstarch_plan,
    )
    plan.parallelize(model, parallel_config)
    context = plan.materialize(device, dtype=torch.float32)
    dataset = build_dataset(
        dataset_name,
        dataset_config,
        split,
        vocab_size=vocab_size,
    )
    sampler = context.create_batch_sampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device == "cuda",
        persistent_workers=False,
    )
    loader = context.prepare_dataloader(dataloader)
    context.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=1e-3),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        ),
    )
    return model, context, loader


def train_one_step(device: str | None = None):
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, context, loader = build_training(device=selected)
    batch = next(iter(loader))
    return context.step(
        batch,
        lambda microbatch: model(microbatch["input_ids"]),
        criterion=lambda logits, microbatch: torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), microbatch["labels"].flatten()
        ),
    )


if __name__ == "__main__":
    print(train_one_step())
