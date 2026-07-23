from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import threading
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


def context(max_nodes: int = 1, global_batch_size: int = 4):
    model = torch.nn.Linear(1, 1)
    plan = OobleckParallelizationPlan(
        OobleckConfig(
            global_batch_size=global_batch_size,
            microbatch_size=2,
            max_nodes=max_nodes,
            seed=9,
        )
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
    original = NodeIdentity("local-node", "original", ("127.0.0.1",), ("0",))
    restarted_node = NodeIdentity("local-node", "restarted", ("127.0.0.1",), ("0",))
    restarted = MembershipSnapshot(
        1,
        (restarted_node,),
        (original,),
        (restarted_node,),
    )
    assert ctx.apply_membership(restarted)
    assert ctx._pending_plan.generation == 1
    assert ctx._pending_plan.previous_generation == 0
    assert ctx._pending_plan.plan_checksum
    assert not ctx.apply_membership(restarted)
    assert ctx._generation_control_metrics[1][0] == 0.0

    lease_node = NodeIdentity("local-node", "lease", ("127.0.0.1",), ("0",))
    lease_expired = MembershipSnapshot(
        2,
        (lease_node,),
        (restarted_node,),
        (lease_node,),
        detection_seconds=ctx.config.lease_timeout_s,
    )
    assert ctx.apply_membership(lease_expired)
    assert ctx._generation_control_metrics[2][0] == ctx.config.lease_timeout_s

    bad_node = NodeIdentity("local-node", "bad", ("127.0.0.1",), ("0", "1"))
    invalid = MembershipSnapshot(
        3,
        (bad_node,),
        (lease_node,),
        (bad_node,),
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


def test_initial_prepare_defers_world_initialization_until_activation():
    model = torch.nn.Linear(1, 1)
    initialized = []
    plan = OobleckParallelizationPlan(
        OobleckConfig(global_batch_size=4, microbatch_size=2, max_nodes=1),
        world_initializer=lambda execution_plan: initialized.append(execution_plan.generation),
    )
    plan.parallelize(model, ParallelConfig())

    prepared = plan.prepare("cpu")
    assert initialized == []
    assert prepared.execution_plan.generation == 0

    ctx = prepared.activate()
    try:
        assert initialized == [0]
        assert ctx.generation == 0
    finally:
        ctx.close()


def test_managed_same_id_restart_prepares_before_rendezvous_activation():
    _, ctx = context()
    initialized = []
    ctx.owner_plan.world_initializer = lambda plan: initialized.append(plan.generation)
    newer = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    ctx.enable_control_plane_barrier()
    assert ctx.announce_generation(newer)

    ctx.prepare_generation()
    assert initialized == []
    assert ctx.generation == 0
    assert ctx.prepared_execution_plan.generation == 1
    assert ctx.partition.closed

    ctx.activate_generation()
    assert initialized == [1]
    assert ctx.generation == 1
    ctx.mark_generation_active(1)
    ctx.close()


def test_concurrent_membership_replays_inflight_managed_step_after_active_barrier():
    model, ctx = context()
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)
    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
    )
    ctx.enable_control_plane_barrier()
    batch = next(iter(loader))
    newer = replace(ctx.execution_plan, generation=1, previous_generation=0, plan_checksum="")
    announced = threading.Event()
    result = []
    errors = []
    first_attempt = True

    def criterion(output, microbatch):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            assert ctx.announce_generation(newer)
            announced.set()
        return ((output - microbatch["y"]) ** 2).mean()

    def train():
        try:
            result.append(ctx.step(batch, lambda item: model(item["x"]), criterion=criterion))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=train)
    worker.start()
    assert announced.wait(timeout=5)
    ctx.prepare_generation()
    assert worker.is_alive()
    ctx.activate_generation()
    assert worker.is_alive()
    ctx.mark_generation_active(1)
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    assert len(result) == 1
    assert result[0].attempts == 2
    assert result[0].committed_step == 1
    assert result[0].generation == 1
    assert sampler.committed_cursor == 1
    ctx.close()


def _node(node_id: str, incarnation_id: str | None = None) -> NodeIdentity:
    return NodeIdentity(node_id, incarnation_id or f"{node_id}-1", ("127.0.0.1",), ("0",))


def _membership(
    generation: int,
    node_ids: tuple[str, ...],
    *,
    removed: tuple[str, ...] = (),
    added: tuple[str, ...] = (),
    previous_plan=None,
    detection_seconds: float = 0.0,
):
    return MembershipSnapshot(
        generation,
        tuple(_node(node_id) for node_id in node_ids),
        tuple(_node(node_id) for node_id in removed),
        tuple(_node(node_id) for node_id in added),
        previous_execution_plan=previous_plan,
        detection_seconds=detection_seconds,
    )


def test_execution_plan_round_trip_is_versioned_and_checksummed():
    _, ctx = context()
    serialized = ctx.execution_plan.to_dict()
    restored = type(ctx.execution_plan).from_dict(serialized)
    assert serialized["schema_version"] == 1
    assert restored == ctx.execution_plan
    assert restored.plan_checksum == ctx.execution_plan.plan_checksum
    corrupted = dict(serialized)
    corrupted["plan_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        type(ctx.execution_plan).from_dict(corrupted)
    ctx.close()


def test_batch_sampler_state_round_trip_restores_committed_cursor():
    _, ctx = context()
    sampler = ctx.create_batch_sampler(Samples(), shuffle=True)
    descriptor = sampler.descriptor_at(0)
    sampler.commit(descriptor)
    state = sampler.state_dict()
    restored = ctx.create_batch_sampler(Samples(), shuffle=True)
    restored.load_state_dict(state)
    assert restored.epoch == sampler.epoch
    assert restored.committed_cursor == 1
    assert restored.descriptor_at(1) == sampler.descriptor_at(1)
    ctx.close()


def test_pure_addition_during_step_commits_once_then_blocks_next_step():
    model, ctx = context(max_nodes=2)
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)
    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
    )
    ctx.enable_control_plane_barrier()
    batch = next(iter(loader))
    entered = threading.Event()
    release = threading.Event()
    results = []

    def criterion(output, microbatch):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return ((output - microbatch["y"]) ** 2).mean()

    worker = threading.Thread(
        target=lambda: results.append(
            ctx.step(batch, lambda item: model(item["x"]), criterion=criterion)
        )
    )
    worker.start()
    assert entered.wait(timeout=5)
    addition = _membership(
        1, ("local-node", "node-b"), added=("node-b",), previous_plan=ctx.execution_plan
    )
    assert ctx.apply_membership(addition)
    assert ctx._deferred_addition_plan is not None
    assert ctx._pending_plan is None
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].attempts == 1
    assert results[0].generation == 0
    assert results[0].committed_step == 1
    assert sampler.committed_cursor == 1
    assert ctx._pending_plan.generation == 1
    assert ctx._transition_metadata[1] == ((), (("node-b", "node-b-1"),), True, 1)

    blocked = threading.Thread(
        target=lambda: ctx.step(
            next(iter(loader)),
            lambda item: model(item["x"]),
            criterion=lambda output, item: output.sum(),
        )
    )
    blocked.start()
    blocked.join(timeout=0.1)
    assert blocked.is_alive()
    ctx._control_plane_managed = False
    ctx._pending_plan = None
    with ctx._generation_condition:
        ctx._generation_condition.notify_all()
    blocked.join(timeout=5)
    ctx.close()


def test_multiple_additions_coalesce_and_failure_supersedes_graceful_cutover():
    model, ctx = context(max_nodes=3, global_batch_size=6)
    dataset = Samples()
    sampler = ctx.create_batch_sampler(dataset, shuffle=False)
    loader = ctx.prepare_dataloader(DataLoader(dataset, batch_sampler=sampler))
    ctx.configure_optimization(
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01)
    )
    ctx.enable_control_plane_barrier()
    batch = next(iter(loader))
    entered = threading.Event()
    release = threading.Event()
    results = []

    def criterion(output, microbatch):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return ((output - microbatch["y"]) ** 2).mean()

    worker = threading.Thread(
        target=lambda: results.append(
            ctx.step(batch, lambda item: model(item["x"]), criterion=criterion)
        )
    )
    worker.start()
    assert entered.wait(timeout=5)
    active = ctx.execution_plan
    assert ctx.apply_membership(
        _membership(1, ("local-node", "node-b"), added=("node-b",), previous_plan=active)
    )
    assert ctx.apply_membership(
        _membership(
            2,
            ("local-node", "node-b", "node-c"),
            added=("node-b", "node-c"),
            previous_plan=active,
        )
    )
    assert ctx._deferred_addition_plan.generation == 2
    assert ctx.apply_membership(
        _membership(
            3,
            ("local-node",),
            removed=("node-b", "node-c"),
            added=("node-b", "node-c"),
            previous_plan=active,
        )
    )
    assert ctx._deferred_addition_plan is None
    assert ctx._pending_plan.generation == 3
    ctx.owner_plan.world_initializer = None
    release.set()
    ctx.prepare_generation()
    ctx.activate_generation()
    ctx.mark_generation_active(3)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].attempts == 2
    assert results[0].generation == 3
    assert results[0].committed_step == 1
    assert ctx._transition_metadata[3] == (
        (("node-b", "node-b-1"), ("node-c", "node-c-1")),
        (("node-b", "node-b-1"), ("node-c", "node-c-1")),
        False,
        0,
    )
    ctx.close()


def test_idle_pure_addition_promotes_immediately_and_is_classified_graceful():
    from oobleck.elastic.membership import is_pure_addition

    _, ctx = context(max_nodes=3, global_batch_size=6)
    snapshot = _membership(
        1, ("local-node", "node-b"), added=("node-b",), previous_plan=ctx.execution_plan
    )
    assert is_pure_addition(snapshot, ctx.execution_plan)
    assert not is_pure_addition(
        _membership(1, ("local-node",), previous_plan=ctx.execution_plan),
        ctx.execution_plan,
    )
    assert not is_pure_addition(
        _membership(
            1,
            ("local-node", "node-b"),
            removed=("local-node",),
            added=("node-b",),
            previous_plan=ctx.execution_plan,
        ),
        ctx.execution_plan,
    )
    assert not is_pure_addition(
        _membership(
            1,
            ("node-b",),
            removed=("local-node",),
            added=("node-b",),
            previous_plan=ctx.execution_plan,
        ),
        ctx.execution_plan,
    )
    assert ctx.apply_membership(snapshot)
    assert ctx._deferred_addition_plan is None
    assert ctx._pending_plan.generation == 1
    assert ctx._transition_metadata[1] == ((), (("node-b", "node-b-1"),), True, 0)
    assert ctx.apply_membership(
        _membership(
            2,
            ("local-node", "node-b", "node-c"),
            added=("node-b", "node-c"),
            previous_plan=ctx.execution_plan,
        )
    )
    assert ctx._pending_plan.generation == 2
    assert ctx._transition_metadata[2] == (
        (),
        (("node-b", "node-b-1"), ("node-c", "node-c-1")),
        True,
        0,
    )
    ctx.close()
