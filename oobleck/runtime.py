"""Public runtime with committed-state generation replacement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from oobleck.distributed import destroy_process_group_universe
from oobleck.elastic.membership import MembershipSnapshot
from oobleck.recovery import capture_context_state, restore_context_state
from oobleck.runtime_base import *  # noqa: F403
from oobleck.runtime_base import OobleckParallelContext
from oobleck.topology import activate_gradient_synchronizer


_base_init = OobleckParallelContext.__init__
_base_configure_optimization = OobleckParallelContext.configure_optimization
_base_step = OobleckParallelContext.step
_base_close = OobleckParallelContext.close


@dataclass(frozen=True, slots=True)
class GenerationTransitionMetrics:
    generation: int
    snapshot_seconds: float
    teardown_seconds: float
    compile_seconds: float
    world_initialization_seconds: float
    activation_seconds: float
    recovery_seconds: float
    total_seconds: float
    superseded: bool = False
    detection_seconds: float = 0.0
    configuration_planning_seconds: float = 0.0
    state_planning_seconds: float = 0.0
    state_transfer_seconds: float = 0.0
    straggler_round_seconds: float = 0.0
    source_scheduling_error_bytes: int = 0


def _init_with_recovery(self: OobleckParallelContext, *args: Any, **kwargs: Any) -> None:
    _base_init(self, *args, **kwargs)
    self._optimizer_factory: (
        Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None
    ) = None
    self._scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None
    self.last_recovery_report = None
    self._control_plane_managed = False
    self._control_active_generation = self.generation
    self.recovery_history: list[GenerationTransitionMetrics] = []
    self._generation_control_metrics: dict[int, tuple[float, float]] = {}
    self._heterogeneous_gradient_sync = (
        None
        if self._needs_survivor_recovery
        else activate_gradient_synchronizer(
            self.model,
            self.partition.manifest,
            self.execution_plan,
            microbatch_size=self.config.microbatch_size,
        )
    )


def _configure_optimization(
    self: OobleckParallelContext,
    *,
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
    scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
    scaler: Any = None,
) -> None:
    _base_configure_optimization(
        self,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scaler=scaler,
    )
    self._optimizer_factory = optimizer_factory
    self._scheduler_factory = scheduler_factory


def _world_initialized() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except (ImportError, RuntimeError):
        return False


def _retire_world(self: OobleckParallelContext) -> None:
    if self._heterogeneous_gradient_sync is not None:
        self._heterogeneous_gradient_sync.close()
        self._heterogeneous_gradient_sync = None
    self.partition.close()
    if _world_initialized():
        destroy_process_group_universe()


def _activate_latest_generation(self: OobleckParallelContext) -> None:
    """Replace WORLD and recover only the last committed training state.

    The snapshot is captured once.  If a newer membership arrives during
    preparation or activation, the partially created generation is retired and
    the same committed snapshot is applied to the newest complete plan.
    """

    if self._pending_plan is None:
        return
    transition_started = time.perf_counter()
    snapshot_started = transition_started
    snapshot = capture_context_state(self)
    snapshot_seconds = time.perf_counter() - snapshot_started
    for loader in self._loaders:
        loader.invalidate_prefetch()

    while self._pending_plan is not None:
        target = self._pending_plan
        self._pending_plan = None
        detection_seconds, configuration_planning_seconds = self._generation_control_metrics.pop(
            target.generation, (0.0, 0.0)
        )
        attempt_started = time.perf_counter()
        teardown_started = attempt_started
        _retire_world(self)
        teardown_seconds = time.perf_counter() - teardown_started

        # Compilation is deliberately process-group independent and occurs
        # after the old universe is gone but before the replacement is created.
        compile_started = time.perf_counter()
        compiled = self.owner_plan.compile(target)
        compile_seconds = time.perf_counter() - compile_started
        world_started = time.perf_counter()
        if self.owner_plan.world_initializer is not None:
            self.owner_plan.world_initializer(target)
        world_seconds = time.perf_counter() - world_started
        activation_started = time.perf_counter()
        activated = compiled.activate(self.device, self.dtype)
        activation_seconds = time.perf_counter() - activation_started

        # A complete newer snapshot supersedes this generation before any
        # committed state is installed or the generation is made visible.
        if self._pending_plan is not None and self._pending_plan.generation > target.generation:
            activated.close()
            if _world_initialized():
                destroy_process_group_universe()
            self.recovery_history.append(
                GenerationTransitionMetrics(
                    target.generation,
                    snapshot_seconds,
                    teardown_seconds,
                    compile_seconds,
                    world_seconds,
                    activation_seconds,
                    0.0,
                    time.perf_counter() - attempt_started,
                    True,
                    detection_seconds=detection_seconds,
                    configuration_planning_seconds=configuration_planning_seconds,
                )
            )
            continue

        self.compiled = compiled
        self.partition = activated
        self.execution_plan = target
        self.owner_plan.rank = compiled.rank
        self.owner_plan._last_execution_plan = target
        for loader in self._loaders:
            loader.reconfigure(target.instances)
        recovery_started = time.perf_counter()
        self.last_recovery_report = restore_context_state(self, snapshot)
        recovery_seconds = time.perf_counter() - recovery_started
        self._heterogeneous_gradient_sync = activate_gradient_synchronizer(
            self.model,
            self.partition.manifest,
            self.execution_plan,
            microbatch_size=self.config.microbatch_size,
        )

        if self._pending_plan is not None:
            # Recovery completed at a safe boundary, but this generation is
            # already stale.  Reuse the original committed snapshot so an
            # intermediate activation can never become a source of truth.
            self.recovery_history.append(
                GenerationTransitionMetrics(
                    target.generation,
                    snapshot_seconds,
                    teardown_seconds,
                    compile_seconds,
                    world_seconds,
                    activation_seconds,
                    recovery_seconds,
                    time.perf_counter() - attempt_started,
                    True,
                    detection_seconds=detection_seconds,
                    configuration_planning_seconds=configuration_planning_seconds,
                    state_planning_seconds=self.last_recovery_report.planning_seconds,
                    state_transfer_seconds=self.last_recovery_report.transfer_seconds,
                    straggler_round_seconds=self.last_recovery_report.straggler_round_seconds,
                    source_scheduling_error_bytes=(
                        self.last_recovery_report.source_scheduling_error_bytes
                    ),
                )
            )
            continue
        self.recovery_history.append(
            GenerationTransitionMetrics(
                target.generation,
                snapshot_seconds,
                teardown_seconds,
                compile_seconds,
                world_seconds,
                activation_seconds,
                recovery_seconds,
                time.perf_counter() - transition_started,
                detection_seconds=detection_seconds,
                configuration_planning_seconds=configuration_planning_seconds,
                state_planning_seconds=self.last_recovery_report.planning_seconds,
                state_transfer_seconds=self.last_recovery_report.transfer_seconds,
                straggler_round_seconds=self.last_recovery_report.straggler_round_seconds,
                source_scheduling_error_bytes=(
                    self.last_recovery_report.source_scheduling_error_bytes
                ),
            )
        )
        break


def _apply_membership(self: OobleckParallelContext, snapshot: MembershipSnapshot) -> bool:
    """Convert one complete control-plane snapshot into a pending generation."""

    newest_generation = max(
        self.generation,
        self._pending_plan.generation if self._pending_plan is not None else -1,
    )
    if snapshot.generation <= newest_generation:
        return False
    tensor_parallel_size = int(getattr(self.owner_plan.parallel_config, "tensor_parallel_size"))
    invalid = [
        node.agent_id for node in snapshot.nodes if len(node.gpu_ids) != tensor_parallel_size
    ]
    if invalid:
        raise ValueError(
            f"membership nodes {invalid} do not match fixed TP width {tensor_parallel_size}"
        )
    coordinator = min(snapshot.nodes, key=lambda node: node.agent_id)
    self.owner_plan.set_rendezvous_address(coordinator.addresses[0])
    planning_started = time.perf_counter()
    self.owner_plan.set_membership(
        tuple(node.agent_id for node in snapshot.nodes), snapshot.generation
    )
    execution_plan = self.owner_plan.build_execution_plan()
    planning_seconds = time.perf_counter() - planning_started
    detection_seconds = (
        self.config.lease_timeout_s
        if any(reason.startswith("lease-expired:") for reason in snapshot.reasons)
        else 0.0
    )
    self._generation_control_metrics[snapshot.generation] = (
        detection_seconds,
        planning_seconds,
    )
    return self.announce_generation(execution_plan)


def _enable_control_plane_barrier(self: OobleckParallelContext) -> None:
    self._control_plane_managed = True
    self._control_active_generation = self.generation


def _prepare_generation(self: OobleckParallelContext) -> None:
    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before preparation")
    _activate_latest_generation(self)


def _mark_generation_active(self: OobleckParallelContext, generation: int) -> None:
    if generation != self.generation:
        raise RuntimeError(
            f"cannot activate generation {generation}; prepared generation is {self.generation}"
        )
    self._control_active_generation = generation


def _step_with_control_barrier(self: OobleckParallelContext, *args: Any, **kwargs: Any) -> Any:
    if self._control_plane_managed and (
        self._pending_plan is not None or self._control_active_generation != self.generation
    ):
        raise RuntimeError(
            "generation is prepared but not active through the CPU control-plane barrier"
        )
    return _base_step(self, *args, **kwargs)


def _recover_from_survivors(self: OobleckParallelContext) -> None:
    """Bootstrap a joining worker that owns no pre-generation state."""

    if not self._needs_survivor_recovery:
        raise RuntimeError("materialize with recover_from_survivors=True for a joining worker")
    self.last_recovery_report = restore_context_state(self, None)
    self._heterogeneous_gradient_sync = activate_gradient_synchronizer(
        self.model,
        self.partition.manifest,
        self.execution_plan,
        microbatch_size=self.config.microbatch_size,
    )
    self._needs_survivor_recovery = False


def _close_with_topology(self: OobleckParallelContext) -> None:
    if self._heterogeneous_gradient_sync is not None:
        self._heterogeneous_gradient_sync.close()
        self._heterogeneous_gradient_sync = None
    _base_close(self)
    if _world_initialized():
        destroy_process_group_universe()


OobleckParallelContext.__init__ = _init_with_recovery
OobleckParallelContext.configure_optimization = _configure_optimization
OobleckParallelContext._activate_latest_generation = _activate_latest_generation
OobleckParallelContext.apply_membership = _apply_membership
OobleckParallelContext.enable_control_plane_barrier = _enable_control_plane_barrier
OobleckParallelContext.prepare_generation = _prepare_generation
OobleckParallelContext.mark_generation_active = _mark_generation_active
OobleckParallelContext.step = _step_with_control_barrier
OobleckParallelContext.recover_from_survivors = _recover_from_survivors
OobleckParallelContext.close = _close_with_topology
