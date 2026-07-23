"""Offline-friendly training setup using Oobleck's public DataLoader path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from oobleck import (
    OobleckConfig,
    OobleckParallelizationPlan,
    build_runtime_compatibility,
)
from oobleck.types import stable_rank_map

if TYPE_CHECKING:
    from oobleck.elastic import MembershipSnapshot


class FakeTextDataset(Dataset):
    def __init__(
        self, samples: int = 128, sequence_length: int = 16, vocab_size: int = 256
    ) -> None:
        generator = torch.Generator().manual_seed(7)
        self.tokens = torch.randint(vocab_size, (samples, sequence_length), generator=generator)
        self.oobleck_fingerprint = (
            f"fake-text:v1:samples={samples}:sequence={sequence_length}:vocab={vocab_size}:seed=7"
        )

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


def _build_model(
    model_backend: str, tensor_parallel_size: int, world_size: int
) -> tuple[torch.nn.Module, object, object | None, int]:
    if tensor_parallel_size < 1 or world_size < tensor_parallel_size:
        raise ValueError("tensor_parallel_size and world_size are inconsistent")
    if model_backend == "tiny":
        return (
            TinyLanguageModel(),
            ExampleParallelConfig(tensor_parallel_size=tensor_parallel_size),
            None,
            256,
        )
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
    cornstarch_plan = ParallelizationPlan(global_ranks=range(world_size))
    return (
        model,
        ParallelConfig(tensor_parallel_size=tensor_parallel_size),
        cornstarch_plan,
        64,
    )


def prepare_training(
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    device: str = "cuda",
    model_backend: str = "tiny",
    membership_snapshot: "MembershipSnapshot | None" = None,
    local_node_id: str | None = None,
    local_tp_lane: int = 0,
    max_nodes: int = 1,
    rendezvous_port: int = 29500,
    distributed_backend: str = "auto",
):
    nodes = (
        tuple(sorted(node.agent_id for node in membership_snapshot.nodes))
        if membership_snapshot is not None
        else ()
    )
    tensor_parallel_size = (
        len(membership_snapshot.nodes[0].gpu_ids) if membership_snapshot is not None else 1
    )
    if membership_snapshot is not None:
        widths = {len(node.gpu_ids) for node in membership_snapshot.nodes}
        if widths != {tensor_parallel_size}:
            raise ValueError("every membership node must contribute the fixed TP width")
        if local_node_id not in nodes:
            raise ValueError("local_node_id must identify a node in the membership snapshot")
        if not 0 <= local_tp_lane < tensor_parallel_size:
            raise ValueError("local_tp_lane is outside the fixed TP width")
        if len(nodes) > max_nodes:
            raise ValueError("membership exceeds configured max_nodes")
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rank_map = dict(stable_rank_map(nodes, tensor_parallel_size)) if nodes else {}
    rank = rank_map[local_node_id][local_tp_lane] if local_node_id is not None else 0
    world_size = max(1, len(nodes) * tensor_parallel_size)
    model, parallel_config, cornstarch_plan, vocab_size = _build_model(
        model_backend, tensor_parallel_size, world_size
    )
    dataset = build_dataset(
        dataset_name,
        dataset_config,
        split,
        vocab_size=vocab_size,
    )
    compatibility = build_runtime_compatibility(
        model,
        dataset,
        model_identity=f"{model_backend}:synthetic-v1:seed={seed}",
        preprocessing_fingerprint="token-shift-v1",
    )
    plan = OobleckParallelizationPlan(
        OobleckConfig(
            global_batch_size=8,
            microbatch_size=2,
            max_nodes=max_nodes,
            seed=seed,
            rendezvous_port=rendezvous_port,
            distributed_backend=distributed_backend,
        ),
        node_ids=nodes,
        rank=rank,
        cornstarch_plan=cornstarch_plan,
        compatibility=compatibility,
    )
    if membership_snapshot is not None:
        plan.set_membership(nodes, membership_snapshot.generation)
        coordinator = min(membership_snapshot.nodes, key=lambda node: node.agent_id)
        plan.set_rendezvous_address(coordinator.addresses[0])
    plan.parallelize(model, parallel_config)
    previous_plan = (
        membership_snapshot.previous_execution_plan
        if membership_snapshot is not None
        else None
    )
    if previous_plan is not None:
        plan._last_execution_plan = previous_plan
    previous_nodes = (
        {node_id for node_id, _ in previous_plan.rank_map}
        if previous_plan is not None
        else set()
    )
    joining = previous_plan is not None and local_node_id not in previous_nodes
    prepared = plan.prepare(
        device, dtype=torch.float32, recover_from_survivors=joining
    )
    return model, prepared, dataset


def configure_training(model, context, dataset, *, device: str):
    """Create data and optimization state only after the generation WORLD is active."""

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


def build_training(**kwargs):
    """Compatibility wrapper that prepares and activates a standalone runtime."""

    model, prepared, dataset = prepare_training(**kwargs)
    context = prepared.activate()
    return configure_training(model, context, dataset, device=str(prepared.device))


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
