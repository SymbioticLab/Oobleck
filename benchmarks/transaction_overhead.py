"""Machine-readable steady-state and transactional recovery benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from oobleck import OobleckConfig, OobleckParallelizationPlan


class LinearDataset(Dataset):
    def __init__(self, samples: int) -> None:
        self.x = torch.arange(samples, dtype=torch.float32).unsqueeze(1) / samples
        self.y = 2 * self.x + 0.5

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: None = None
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1


def run(*, steady_steps: int = 4) -> dict[str, object]:
    if steady_steps < 1:
        raise ValueError("steady_steps must be positive")
    torch.manual_seed(11)
    initial = torch.nn.Linear(1, 1).state_dict()
    dataset = LinearDataset((steady_steps + 1) * 4)
    warmup = torch.nn.Linear(1, 1)
    warmup_optimizer = torch.optim.SGD(warmup.parameters(), lr=0.05)
    warmup_optimizer.zero_grad(set_to_none=True)
    warmup_loss = torch.nn.functional.mse_loss(warmup(dataset.x[:4]), dataset.y[:4])
    warmup_loss.backward()
    warmup_optimizer.step()

    reference = torch.nn.Linear(1, 1)
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    baseline_durations = []
    for step in range(steady_steps + 1):
        batch = {key: value[step * 4 : (step + 1) * 4] for key, value in vars(dataset).items()}
        started = time.perf_counter()
        reference_optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(reference(batch["x"]), batch["y"])
        loss.backward()
        reference_optimizer.step()
        baseline_durations.append(time.perf_counter() - started)

    model = torch.nn.Linear(1, 1)
    model.load_state_dict(initial)
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=4, microbatch_size=2, max_nodes=1, seed=11)
    )
    plan.parallelize(model, ParallelConfig())
    context = plan.materialize("cpu")
    sampler = context.create_batch_sampler(dataset, shuffle=False)
    loader = context.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
    context.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.05)
    )
    iterator = iter(loader)
    steady_durations = []
    for _ in range(steady_steps):
        batch = next(iterator)
        started = time.perf_counter()
        context.step(
            batch,
            lambda microbatch: model(microbatch["x"]),
            criterion=lambda output, microbatch: torch.nn.functional.mse_loss(
                output, microbatch["y"]
            ),
        )
        steady_durations.append(time.perf_counter() - started)

    recovery_batch = next(iterator)
    newer = replace(
        context.execution_plan,
        generation=context.generation + 1,
        previous_generation=context.generation,
        plan_checksum="",
    )
    announced = False

    def recovery_criterion(output, microbatch):
        nonlocal announced
        if not announced:
            announced = True
            context.announce_generation(newer)
        return torch.nn.functional.mse_loss(output, microbatch["y"])

    recovery_started = time.perf_counter()
    recovery_result = context.step(
        recovery_batch,
        lambda microbatch: model(microbatch["x"]),
        criterion=recovery_criterion,
    )
    recovery_seconds = time.perf_counter() - recovery_started
    equivalent = all(
        torch.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in zip(model.parameters(), reference.parameters())
    )
    baseline_mean = sum(baseline_durations[:steady_steps]) / steady_steps
    steady_mean = sum(steady_durations) / steady_steps
    result = {
        "schema_version": 1,
        "scenario": {
            "device": "cpu",
            "steady_steps": steady_steps,
            "global_batch_size": 4,
            "microbatch_size": 2,
        },
        "baseline_step_seconds": baseline_mean,
        "oobleck_steady_step_seconds": steady_mean,
        "steady_state_overhead_ratio": steady_mean / baseline_mean,
        "recovery_step_seconds": recovery_seconds,
        "recovery_extra_seconds": max(0.0, recovery_seconds - steady_mean),
        "recovery_attempts": recovery_result.attempts,
        "committed_step": recovery_result.committed_step,
        "replayed_sample_indices": list(recovery_batch.sample_indices),
        "numerically_equivalent": equivalent,
    }
    context.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(run(steady_steps=arguments.steady_steps), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)


if __name__ == "__main__":
    main()
