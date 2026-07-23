from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from oobleck import OobleckConfig, OobleckParallelizationPlan
from oobleck.elastic import MembershipSnapshot, NodeIdentity


class Samples(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index)]), "y": torch.tensor([float(index * 2)])}


class Stream(IterableDataset):
    def __iter__(self):
        yield {"x": 1}


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: None = None
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1


def context():
    model = torch.nn.Linear(1, 1)
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=4, microbatch_size=2, max_nodes=1, seed=9)
    )
    plan.parallelize(model, ParallelConfig())
    return model, plan.materialize("cpu")


def test_sampler_does_not_advance_until_commit_and_replays_indices():
    _, ctx = context()
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=True)
    first = list(next(iter(sampler)))
    assert list(next(iter(sampler))) == first
    assert sampler.committed_cursor == 0


def test_dataset_contract_rejects_iterable_and_dataset_dict():
    _, ctx = context()
    with pytest.raises(TypeError, match="map-style"):
        ctx.create_batch_sampler(Stream(), shuffle=False)
    datasets = pytest.importorskip("datasets")
    value = datasets.DatasetDict({"train": datasets.Dataset.from_dict({"x": [1, 2]})})
    with pytest.raises(TypeError, match="Select a split"):
        ctx.create_batch_sampler(value, shuffle=False)


def test_generation_change_discards_gradients_and_replays_once():
    model, ctx = context()
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)
    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
    scheduler_steps = []

    class Scheduler:
        def step(self):
            scheduler_steps.append(1)

    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        scheduler_factory=lambda optimizer: Scheduler(),
    )
    batch = next(iter(loader))
    newer = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    announced = False

    def criterion(output, microbatch):
        nonlocal announced
        if not announced:
            announced = True
            ctx.announce_generation(newer)
        return ((output - microbatch["y"]) ** 2).mean()

    result = ctx.step(batch, lambda microbatch: model(microbatch["x"]), criterion=criterion)
    assert result.attempts == 2
    assert result.committed_step == 1
    assert result.generation == 1
    assert sampler.committed_cursor == 1
    assert len(scheduler_steps) == 1


def test_generation_change_restores_adam_scheduler_and_scaler_state():
    model, ctx = context()

    class Scheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.count = 0

        def step(self):
            self.count += 1

        def state_dict(self):
            return {"count": self.count}

        def load_state_dict(self, state):
            self.count = state["count"]

    class Scaler:
        def __init__(self):
            self.growth = 8

        def state_dict(self):
            return {"growth": self.growth}

        def load_state_dict(self, state):
            self.growth = state["growth"]

    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=0.03),
        scheduler_factory=Scheduler,
        scaler=Scaler(),
    )
    model(torch.tensor([[2.0]])).sum().backward()
    ctx.optimizer.step()
    ctx.optimizer.zero_grad(set_to_none=True)
    ctx.scheduler.step()
    ctx.committed_step = 1
    expected_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    expected_slots = {
        name: {slot: value.detach().clone() for slot, value in ctx.optimizer.state[param].items()}
        for name, param in model.named_parameters()
    }
    old_optimizer = ctx.optimizer

    newer = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    assert ctx.announce_generation(newer)
    ctx._activate_latest_generation()

    assert ctx.optimizer is not old_optimizer
    assert ctx.committed_step == 1
    assert ctx.scheduler.count == 1
    assert ctx.scaler.growth == 8
    assert ctx.last_recovery_report.schedule.transfers == ()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name])
        for slot, expected in expected_slots[name].items():
            torch.testing.assert_close(ctx.optimizer.state[parameter][slot], expected)


def test_newer_generation_supersedes_in_progress_activation():
    _, ctx = context()
    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
    )
    first = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    second = replace(ctx.execution_plan, generation=2, previous_generation=1, plan_checksum="")
    initialized = []

    def initialize(target):
        initialized.append(target.generation)
        if target.generation == 1:
            ctx.announce_generation(second)

    ctx.owner_plan.world_initializer = initialize
    ctx.announce_generation(first)
    ctx._activate_latest_generation()

    assert initialized == [1, 2]
    assert ctx.generation == 2
    assert ctx.partition.closed is False


def test_heterogeneous_pipelines_execute_disjoint_global_microbatches():
    dataset = Samples()
    template = __import__("oobleck").PipelineTemplate("one", ((0, 1),), 1, 1, 1)
    observed = []
    for rank in (0, 1):
        model = torch.nn.Linear(1, 1)
        plan = OobleckParallelizationPlan(
            OobleckConfig(global_batch_size=8, microbatch_size=2, max_nodes=2),
            templates=(template,),
            node_ids=("node-a", "node-b"),
            rank=rank,
        )
        plan.parallelize(model, ParallelConfig())
        ctx = plan.materialize("cpu")
        sampler = ctx.create_batch_sampler(dataset, shuffle=False)
        loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
        ctx.configure_optimization(
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
        )
        batch = next(iter(loader))
        calls = []

        def execute(microbatch):
            calls.extend(int(item) for item in microbatch["x"].flatten())
            return model(microbatch["x"])

        ctx.step(batch, execute, criterion=lambda output, item: output.sum())
        pipeline_id = ctx.execution_plan.rank_local_stage(rank).pipeline_id
        assignment = next(
            item for item in batch.descriptor.assignments if item.pipeline_id == pipeline_id
        )
        assert tuple(calls) == assignment.sample_indices
        observed.extend(calls)

    assert sorted(observed) == list(range(8))
    assert len(observed) == len(set(observed))


def test_complete_membership_snapshot_announces_deterministic_execution_plan():
    _, ctx = context()
    replacement = MembershipSnapshot(
        1,
        (NodeIdentity("local-node", "replacement", ("127.0.0.1",), ("0",)),),
        ("replacement:local-node",),
    )
    assert ctx.apply_membership(replacement)
    assert ctx._pending_plan.generation == 1
    assert ctx._pending_plan.previous_generation == 0
    assert ctx._pending_plan.plan_checksum
    assert not ctx.apply_membership(replacement)

    invalid = MembershipSnapshot(
        2,
        (NodeIdentity("local-node", "bad", ("127.0.0.1",), ("0", "1")),),
        ("replacement:local-node",),
    )
    with pytest.raises(ValueError, match="fixed TP width"):
        ctx.apply_membership(invalid)


def test_initial_meta_partition_runs_checkpoint_initializer_before_training():
    class CheckpointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(3, device="meta"))
            self.register_buffer("running", torch.empty(2, device="meta"))
            self.initialized = False

        def initialize_checkpoint(self):
            with torch.no_grad():
                self.weight.fill_(7)
                self.running.fill_(11)
            self.initialized = True

        def forward(self, value):
            return value * self.weight[0]

    model = CheckpointModel()
    plan = OobleckParallelizationPlan(OobleckConfig(global_batch_size=2, microbatch_size=1))
    plan.parallelize(model, ParallelConfig())
    context = plan.materialize("cpu")

    assert model.initialized
    assert not model.weight.is_meta and not model.running.is_meta
    torch.testing.assert_close(model.weight, torch.full((3,), 7.0))
    torch.testing.assert_close(model.running, torch.full((2,), 11.0))
    context.close()


def test_hugging_face_dataset_replays_through_standard_dataloader():
    datasets = pytest.importorskip("datasets")
    model, ctx = context()
    dataset = datasets.Dataset.from_dict(
        {
            "x": [float(index) for index in range(16)],
            "y": [float(index * 2) for index in range(16)],
        }
    )
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)

    def collate(rows):
        return {
            key: torch.tensor([[row[key]] for row in rows], dtype=torch.float32)
            for key in ("x", "y")
        }

    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler, collate_fn=collate))
    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
    )
    batch = next(iter(loader))
    newer = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    announced = False

    def criterion(output, microbatch):
        nonlocal announced
        if not announced:
            announced = True
            ctx.announce_generation(newer)
        return ((output - microbatch["y"]) ** 2).mean()

    result = ctx.step(batch, lambda microbatch: model(microbatch["x"]), criterion=criterion)
    assert result.attempts == 2
    assert sampler.committed_cursor == 1
    assert len(set(batch.sample_indices)) == len(batch.sample_indices) == 4
    ctx.close()


def test_prepare_dataloader_rejects_non_oobleck_and_persistent_worker_configs():
    _, ctx = context()
    dataset = Samples()
    with pytest.raises(TypeError, match="OobleckBatchSampler"):
        ctx.prepare_dataloader(DataLoader(dataset, batch_size=4))
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)
    persistent = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=1,
        persistent_workers=True,
    )
    with pytest.raises(ValueError, match="persistent_workers"):
        ctx.prepare_dataloader(persistent)
    ctx.close()


def test_collator_may_return_an_explicit_microbatch_list():
    _, ctx = context()
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)

    def collate(rows):
        return [
            {key: torch.stack([row[key] for row in rows[start : start + 2]]) for key in ("x", "y")}
            for start in (0, 2)
        ]

    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler, collate_fn=collate))
    batch = next(iter(loader))
    assert len(batch.microbatches) == 2
    assert [item["x"].flatten().tolist() for item in batch.microbatches] == [
        [0.0, 1.0],
        [2.0, 3.0],
    ]
    ctx.close()
