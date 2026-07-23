"""Public Oobleck LLM example with standalone and agent-managed execution."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pretrain_llm_base import (
    ExampleParallelConfig,
    FakeTextDataset,
    TinyLanguageModel,
    build_dataset,
    build_training,
    configure_training,
    prepare_training,
)
from oobleck.acceptance import append_metric, resolve_metrics_path, runtime_metric
from oobleck.elastic import LocalWorkerClient


@dataclass(frozen=True)
class ExampleTrainingConfig:
    device: str | None = None
    model_backend: str = "tiny"
    dataset_name: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    max_nodes: int = 16
    rendezvous_port: int = 29500
    distributed_backend: str = "auto"
    steps: int = 1
    step_delay_s: float = 0.0
    metrics_output: Path | None = None

    def __post_init__(self) -> None:
        if self.steps < 1 or self.step_delay_s < 0:
            raise ValueError("steps must be positive and step_delay_s must be non-negative")


def _one_step(model, context, loader, selected: str, model_backend: str):
    batch = next(iter(loader))
    if model_backend == "cornstarch":
        from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture

        execution = CornstarchExecutionPlan()
        merged = execution.merge_modality_encoder_outputs(
            language_model=model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={},
            encoder_outputs={},
            language_model_inputs={},
        )
        output = execution.run_language_model(module=model, inputs=merged)
        return context.step(
            batch,
            lambda microbatch: output.execute(
                inputs={
                    key: value.to(selected, non_blocking=True) for key, value in microbatch.items()
                }
            ),
            criterion=lambda language_model_output, _: language_model_output.loss,
        )
    if model_backend != "tiny":
        raise ValueError("model_backend must be 'tiny' or 'cornstarch'")
    return context.step(
        batch,
        lambda microbatch: model(microbatch["input_ids"].to(selected, non_blocking=True)),
        criterion=lambda logits, microbatch: torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            microbatch["labels"].to(logits.device, non_blocking=True).flatten(),
        ),
    )


def train_one_step(
    device: str | None = None,
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    model_backend: str = "tiny",
    max_nodes: int = 1,
    rendezvous_port: int = 29500,
    distributed_backend: str = "auto",
):
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, context, loader = build_training(
        device=selected,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        model_backend=model_backend,
        max_nodes=max_nodes,
        rendezvous_port=rendezvous_port,
        distributed_backend=distributed_backend,
    )
    try:
        return _one_step(model, context, loader, selected, model_backend)
    finally:
        context.close()


async def train_managed(config: ExampleTrainingConfig):
    """Compile before rendezvous and train only after generation_active."""

    socket_path = os.environ["OOBLECK_LOCAL_WORKER_SOCKET"]
    node_id = os.environ["OOBLECK_NODE_ID"]
    worker_id = os.environ["OOBLECK_WORKER_ID"]
    selected = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    worker = LocalWorkerClient(socket_path, node_id, worker_id)
    await worker.connect()
    snapshot = await worker.receive()
    model, prepared, dataset = prepare_training(
        device=selected,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.split,
        model_backend=config.model_backend,
        membership_snapshot=snapshot,
        local_node_id=node_id,
        local_tp_lane=int(os.environ["OOBLECK_LOCAL_RANK"]),
        max_nodes=config.max_nodes,
        rendezvous_port=config.rendezvous_port,
        distributed_backend=config.distributed_backend,
    )
    context = await worker.activate_prepared(prepared, snapshot)
    model, context, loader = configure_training(model, context, dataset, device=selected)

    metrics_path = (
        None
        if config.metrics_output is None
        else resolve_metrics_path(config.metrics_output, node_id=node_id, worker_id=worker_id)
    )

    def record(event: str, result=None, *, step_seconds: float = 0.0) -> None:
        if metrics_path is not None:
            append_metric(
                metrics_path,
                runtime_metric(
                    context,
                    event=event,
                    node_id=node_id,
                    worker_id=worker_id,
                    result=result,
                    step_seconds=step_seconds,
                ),
            )

    async def prepare(snapshot) -> None:
        context.prepare_generation()

    async def activated(generation: int) -> None:
        record("generation_active")

    record("generation_active")
    relay_task = asyncio.create_task(
        worker.run_context(context, on_snapshot=prepare, on_active=activated)
    )
    try:
        result = None
        for _ in range(config.steps):
            started = time.perf_counter()
            result = await asyncio.to_thread(
                _one_step, model, context, loader, selected, config.model_backend
            )
            record("step", result, step_seconds=time.perf_counter() - started)
            await asyncio.sleep(config.step_delay_s)
        assert result is not None
        return result
    finally:
        relay_task.cancel()
        await asyncio.gather(relay_task, return_exceptions=True)
        await worker.close()
        context.close()
        record("closed")


def main() -> None:
    import tyro

    config = tyro.cli(ExampleTrainingConfig)
    if "OOBLECK_LOCAL_WORKER_SOCKET" in os.environ:
        result = asyncio.run(train_managed(config))
    else:
        result = train_one_step(
            config.device,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            model_backend=config.model_backend,
            max_nodes=config.max_nodes,
            rendezvous_port=config.rendezvous_port,
            distributed_backend=config.distributed_backend,
        )
    print(
        f"committed_step={result.committed_step} generation={result.generation} "
        f"attempts={result.attempts} loss={result.loss:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ExampleParallelConfig",
    "ExampleTrainingConfig",
    "FakeTextDataset",
    "TinyLanguageModel",
    "build_dataset",
    "build_training",
    "configure_training",
    "prepare_training",
    "train_managed",
    "train_one_step",
]
