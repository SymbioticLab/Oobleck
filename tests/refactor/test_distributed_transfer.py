import torch
import torch.distributed as dist
from types import SimpleNamespace

from oobleck.recovery import capture_context_state, restore_context_state
from oobleck.state import LogicalStateEntry, StateManifest, plan_state_redistribution
from oobleck.state_transfer import execute_transfer_schedule
from oobleck.topology import activate_gradient_synchronizer
from oobleck.types import OobleckExecutionPlan, PipelineInstance, PipelineTemplate
from tests.distributed.distributed_base import GlooDistributedTestBase


class TestDistributedStateTransfer(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_variable_split_all_to_all_reconstructs_cuda_state(self):
        device = torch.device("cuda", self.rank % torch.cuda.device_count())

        def entry(rank):
            return LogicalStateEntry(
                "layer.weight",
                (17,),
                (17,),
                "float32",
                "parameter",
                ("replicate",),
                0,
                rank,
                4,
            )

        schedule = plan_state_redistribution(
            [StateManifest(0, (entry(0),), 4)],
            [StateManifest(1, (entry(1),), 4)],
            chunk_bytes=20,
            alignment=4,
        )
        key = ("layer.weight", "parameter", 0)
        expected = torch.arange(17, dtype=torch.float32, device=device)
        sources = {key: expected} if self.rank == 0 else {}
        destinations = {key: torch.zeros_like(expected)} if self.rank == 1 else {}
        metrics = execute_transfer_schedule(
            schedule,
            rank=self.rank,
            world_size=self.world_size,
            sources=sources,
            destinations=destinations,
            device=device,
        )
        assert metrics.actual_source_bytes == schedule.per_source_bytes
        assert metrics.actual_destination_bytes == schedule.per_destination_bytes
        assert all(duration >= 0 for _, _, duration in metrics.round_durations)
        if self.rank == 1:
            torch.testing.assert_close(destinations[key], expected)

    def test_runtime_recovery_moves_model_and_adam_state_between_cuda_ranks(self):
        assert dist.get_backend() == "gloo"
        device = torch.device("cuda", self.rank % torch.cuda.device_count())
        rank = self.rank

        class LogicalWeight(torch.nn.Module):
            def __init__(self, logical_key):
                super().__init__()
                self.logical_key = logical_key
                self.weight = torch.nn.Parameter(torch.full((5,), 10.0 + rank, device=device))

            def _global_cornstarch_key(self, local_key):
                return self.logical_key if local_key == "weight" else local_key

        old_key = f"layer.{self.rank}.weight"
        new_key = f"layer.{1 - self.rank}.weight"
        model = LogicalWeight(old_key)

        def optimizer_factory(parameters):
            return torch.optim.AdamW(parameters, lr=0.02)

        optimizer = optimizer_factory(model.parameters())
        model.weight.grad = torch.full_like(model.weight, float(self.rank + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        def manifest(logical_key):
            return StateManifest(
                self.rank,
                (
                    LogicalStateEntry(
                        logical_key,
                        (5,),
                        (5,),
                        "torch.float32",
                        "parameter",
                        ("replicate",),
                        0,
                        self.rank,
                        3,
                    ),
                ),
                3,
            )

        context = SimpleNamespace(
            model=model,
            partition=SimpleNamespace(manifest=manifest(old_key)),
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            committed_step=3,
            owner_plan=SimpleNamespace(rank=self.rank),
            compiled=SimpleNamespace(world_size=self.world_size),
            config=SimpleNamespace(state_transfer_chunk_bytes=8, transfer_alignment_bytes=4),
            device=device,
            _optimizer_factory=optimizer_factory,
            _scheduler_factory=None,
            execution_plan=SimpleNamespace(rank_map=(("one-node", (0, 1)),)),
        )
        snapshot = capture_context_state(context)
        source_weight = model.weight.detach().cpu().clone()
        source_exp_avg = optimizer.state[model.weight]["exp_avg"].detach().cpu().clone()
        gathered_weights = [None] * self.world_size
        gathered_moments = [None] * self.world_size
        dist.all_gather_object(gathered_weights, source_weight)
        dist.all_gather_object(gathered_moments, source_exp_avg)

        model.logical_key = new_key
        context.partition.manifest = manifest(new_key)
        report = restore_context_state(context, snapshot)

        assert model.weight.is_cuda
        assert report.schedule.transfers
        assert report.actual_source_bytes == report.source_bytes
        assert report.actual_destination_bytes == report.destination_bytes
        assert report.round_durations
        assert report.link_class_bytes
        assert {link for link, _ in report.link_class_bytes} == {"same-node"}
        torch.testing.assert_close(model.weight.cpu(), gathered_weights[1 - self.rank])
        torch.testing.assert_close(
            context.optimizer.state[model.weight]["exp_avg"].cpu(),
            gathered_moments[1 - self.rank],
        )

    def test_heterogeneous_pipeline_gradients_use_global_sample_weights_on_cuda(self):
        assert dist.get_backend() == "gloo"
        device = torch.device("cuda", self.rank % torch.cuda.device_count())

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3, device=device))

        model = Model()
        model.weight.grad = torch.full_like(model.weight, float(self.rank + 1))
        template = PipelineTemplate("single", ((0, 1),), 1, 1.0, 1.0)
        instances = (
            PipelineInstance("fast", template, ("node-0",), ((0,),), 3),
            PipelineInstance("slow", template, ("node-1",), ((1,),), 1),
        )
        plan = OobleckExecutionPlan(
            0,
            instances,
            (("node-0", (0,)), ("node-1", (1,))),
        )
        manifest = StateManifest(
            self.rank,
            (
                LogicalStateEntry(
                    "weight",
                    (3,),
                    (3,),
                    "torch.float32",
                    "parameter",
                    ("replicate",),
                    0,
                    self.rank,
                    0,
                ),
            ),
            0,
        )
        synchronizer = activate_gradient_synchronizer(model, manifest, plan, microbatch_size=2)
        assert synchronizer is not None
        synchronizer.sync()

        # 6 samples from rank 0 and 2 from rank 1: 1*(6/8) + 2*(2/8).
        assert model.weight.grad.is_cuda
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight.grad, 1.25))
        synchronizer.close()

    def test_added_rank_recovers_parameter_and_optimizer_without_local_snapshot(self):
        assert dist.get_backend() == "gloo"
        device = torch.device("cuda", self.rank % torch.cuda.device_count())

        class StateModel(torch.nn.Module):
            def __init__(self, keys, base):
                super().__init__()
                for offset, key in enumerate(keys):
                    self.register_parameter(
                        key,
                        torch.nn.Parameter(torch.full((4,), base + offset, device=device)),
                    )

        def manifest(keys, committed_step):
            return StateManifest(
                self.rank,
                tuple(
                    LogicalStateEntry(
                        key,
                        (4,),
                        (4,),
                        "torch.float32",
                        "parameter",
                        ("replicate",),
                        0,
                        self.rank,
                        committed_step,
                    )
                    for key in keys
                ),
                committed_step,
            )

        def optimizer_factory(parameters):
            return torch.optim.AdamW(parameters, lr=0.01)

        class Stateful:
            def __init__(self, value):
                self.value = value

            def state_dict(self):
                return {"value": self.value}

            def load_state_dict(self, state):
                self.value = state["value"]

        def scheduler_factory(optimizer):
            return Stateful(0)

        if self.rank == 0:
            source_model = StateModel(("a", "b"), 5.0)
            source_optimizer = optimizer_factory(source_model.parameters())
            for index, parameter in enumerate(source_model.parameters()):
                parameter.grad = torch.full_like(parameter, float(index + 1))
            source_optimizer.step()
            source_optimizer.zero_grad(set_to_none=True)
            context = SimpleNamespace(
                model=source_model,
                partition=SimpleNamespace(manifest=manifest(("a", "b"), 5)),
                optimizer=source_optimizer,
                scheduler=Stateful(7),
                scaler=Stateful(9),
                committed_step=5,
                owner_plan=SimpleNamespace(rank=0),
                compiled=SimpleNamespace(world_size=self.world_size),
                config=SimpleNamespace(state_transfer_chunk_bytes=8, transfer_alignment_bytes=4),
                device=device,
                _optimizer_factory=optimizer_factory,
                _scheduler_factory=scheduler_factory,
                _loaders=[SimpleNamespace(sampler=Stateful(3))],
            )
            snapshot = capture_context_state(context)
            expected = (
                source_model.b.detach().cpu(),
                source_optimizer.state[source_model.b]["exp_avg"].detach().cpu(),
            )
            context.model = StateModel(("a",), -1.0)
            context.partition.manifest = manifest(("a",), 5)
        else:
            target_model = StateModel(("b",), -1.0)
            context = SimpleNamespace(
                model=target_model,
                partition=SimpleNamespace(manifest=manifest(("b",), 0)),
                optimizer=optimizer_factory(target_model.parameters()),
                scheduler=Stateful(0),
                scaler=Stateful(0),
                committed_step=0,
                owner_plan=SimpleNamespace(rank=1),
                compiled=SimpleNamespace(world_size=self.world_size),
                config=SimpleNamespace(state_transfer_chunk_bytes=8, transfer_alignment_bytes=4),
                device=device,
                _optimizer_factory=optimizer_factory,
                _scheduler_factory=scheduler_factory,
                _loaders=[SimpleNamespace(sampler=Stateful(0))],
            )
            snapshot = None
            expected = None

        expected_values = [expected]
        dist.broadcast_object_list(expected_values, src=0)
        report = restore_context_state(context, snapshot)

        assert context.committed_step == 5
        assert context.scheduler.value == 7
        assert context.scaler.value == 9
        assert context._loaders[0].sampler.value == 3
        assert report.total_seconds > 0
        assert report.actual_source_bytes == report.source_bytes
        assert report.round_durations
        if self.rank == 1:
            expected_weight, expected_moment = expected_values[0]
            torch.testing.assert_close(context.model.b.cpu(), expected_weight)
            torch.testing.assert_close(
                context.optimizer.state[context.model.b]["exp_avg"].cpu(),
                expected_moment,
            )
