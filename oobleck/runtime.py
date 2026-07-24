"""Public runtime with committed-state generation replacement."""

from __future__ import annotations

import threading
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
    """Per-phase latency, balancing, and supersession data for one generation."""

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


@dataclass(slots=True)
class _PreparedGenerationTransition:
    """Snapshot and compiled ownership held between control-plane barriers."""

    target: Any
    compiled: Any
    snapshot: Any
    snapshot_seconds: float
    teardown_seconds: float
    compile_seconds: float
    transition_started: float
    attempt_started: float
    detection_seconds: float
    configuration_planning_seconds: float
    superseded_recorded: bool = False


def _init_with_recovery(self: OobleckParallelContext, *args: Any, **kwargs: Any) -> None:
    """Extend the base context with recovery factories, barriers, and topology."""

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
    self._prepared_transition: _PreparedGenerationTransition | None = None
    self._generation_condition = threading.Condition()
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
    """Retain optimizer factories so replacement partitions can rebuild state."""

    _base_configure_optimization(
        self,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scaler=scaler,
    )
    self._optimizer_factory = optimizer_factory
    self._scheduler_factory = scheduler_factory


def _world_initialized() -> bool:
    """Safely query WORLD without requiring distributed support to be importable."""

    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except (ImportError, RuntimeError):
        return False


def _retire_world(self: OobleckParallelContext) -> None:
    """Close logical topology and partition before destroying every process group."""

    if self._heterogeneous_gradient_sync is not None:
        self._heterogeneous_gradient_sync.close()
        self._heterogeneous_gradient_sync = None
    self.partition.close()
    if _world_initialized():
        destroy_process_group_universe()


def _record_superseded(
    self: OobleckParallelContext,
    transition: _PreparedGenerationTransition,
    *,
    world_seconds: float = 0.0,
    activation_seconds: float = 0.0,
    recovery_seconds: float = 0.0,
) -> None:
    """Record one superseded generation attempt without double-counting recovery work.

    The transition remembers whether it was already emitted because supersession can be observed at
    compile, activation, or post-recovery boundaries. Completed phase durations and any available state
    transfer metrics are preserved, while the record is marked superseded so acceptance analysis can
    distinguish abandoned work from the generation that ultimately became active.
    """

    if transition.superseded_recorded:
        return
    transition.superseded_recorded = True
    report = self.last_recovery_report if recovery_seconds else None
    self.recovery_history.append(
        GenerationTransitionMetrics(
            transition.target.generation,
            transition.snapshot_seconds,
            transition.teardown_seconds,
            transition.compile_seconds,
            world_seconds,
            activation_seconds,
            recovery_seconds,
            time.perf_counter() - transition.attempt_started,
            True,
            detection_seconds=transition.detection_seconds,
            configuration_planning_seconds=transition.configuration_planning_seconds,
            state_planning_seconds=0.0 if report is None else report.planning_seconds,
            state_transfer_seconds=0.0 if report is None else report.transfer_seconds,
            straggler_round_seconds=0.0 if report is None else report.straggler_round_seconds,
            source_scheduling_error_bytes=(
                0 if report is None else report.source_scheduling_error_bytes
            ),
        )
    )


def _prepare_latest_generation(self: OobleckParallelContext) -> bool:
    """Perform the first generation barrier: snapshot, teardown, and compile.

    The first pending proposal clones the last committed state, invalidates loader
    prefetch, and completely retires the old process-group universe. Superseding
    proposals reuse that same committed snapshot rather than snapshotting partial
    recovery state, record the abandoned attempt, and compile only the newest plan.
    The resulting transition is process-group-free and safe to checksum before the
    master publishes rendezvous.
    """

    existing = self._prepared_transition
    if self._pending_plan is None:
        return existing is not None

    if existing is None:
        transition_started = time.perf_counter()
        snapshot_started = transition_started
        snapshot = capture_context_state(self)
        snapshot_seconds = time.perf_counter() - snapshot_started
        for loader in self._loaders:
            loader.invalidate_prefetch()
        teardown_started = time.perf_counter()
        _retire_world(self)
        teardown_seconds = time.perf_counter() - teardown_started
    else:
        transition_started = existing.transition_started
        snapshot = existing.snapshot
        snapshot_seconds = existing.snapshot_seconds
        teardown_seconds = 0.0
        _record_superseded(self, existing)
        self._prepared_transition = None

    while self._pending_plan is not None:
        target = self._pending_plan
        self._pending_plan = None
        detection_seconds, configuration_planning_seconds = self._generation_control_metrics.pop(
            target.generation, (0.0, 0.0)
        )
        attempt_started = time.perf_counter()
        compile_started = attempt_started
        compiled = self.owner_plan.compile(target)
        compile_seconds = time.perf_counter() - compile_started
        transition = _PreparedGenerationTransition(
            target,
            compiled,
            snapshot,
            snapshot_seconds,
            teardown_seconds,
            compile_seconds,
            transition_started,
            attempt_started,
            detection_seconds,
            configuration_planning_seconds,
        )
        if self._pending_plan is not None and self._pending_plan.generation > target.generation:
            _record_superseded(self, transition)
            teardown_seconds = 0.0
            continue
        self._prepared_transition = transition
        return True
    return False


def _activate_prepared_generation(self: OobleckParallelContext) -> bool:
    """Perform the second generation barrier: rendezvous, activation, and recovery.

    Replacement WORLD is initialized before Cornstarch materializes local ownership.
    All ranks then redistribute the shared committed snapshot, rebuild heterogeneous
    gradient groups, and reconfigure attached DataLoaders. A newer proposal at any
    point retires partially initialized resources and returns to preparation using
    the original committed snapshot; only an unsuperseded transition enters recovery
    history as the active generation.
    """

    transition = self._prepared_transition
    if transition is None:
        raise RuntimeError("no prepared generation is waiting for rendezvous")
    target = transition.target
    world_started = time.perf_counter()
    if self.owner_plan.world_initializer is not None:
        self.owner_plan.world_initializer(target)
    world_seconds = time.perf_counter() - world_started
    activation_started = time.perf_counter()
    activated = transition.compiled.activate(self.device, self.dtype)
    activation_seconds = time.perf_counter() - activation_started

    if self._pending_plan is not None and self._pending_plan.generation > target.generation:
        activated.close()
        if _world_initialized():
            destroy_process_group_universe()
        _record_superseded(
            self,
            transition,
            world_seconds=world_seconds,
            activation_seconds=activation_seconds,
        )
        self._prepared_transition = transition
        _prepare_latest_generation(self)
        return False

    self.compiled = transition.compiled
    self.partition = activated
    self.execution_plan = target
    self.owner_plan.rank = transition.compiled.rank
    self.owner_plan._last_execution_plan = target
    for loader in self._loaders:
        loader.reconfigure(target.instances)
    recovery_started = time.perf_counter()
    self.last_recovery_report = restore_context_state(self, transition.snapshot)
    recovery_seconds = time.perf_counter() - recovery_started
    self._heterogeneous_gradient_sync = activate_gradient_synchronizer(
        self.model,
        self.partition.manifest,
        self.execution_plan,
        microbatch_size=self.config.microbatch_size,
    )

    if self._pending_plan is not None:
        _record_superseded(
            self,
            transition,
            world_seconds=world_seconds,
            activation_seconds=activation_seconds,
            recovery_seconds=recovery_seconds,
        )
        _retire_world(self)
        self._prepared_transition = transition
        _prepare_latest_generation(self)
        return False

    self.recovery_history.append(
        GenerationTransitionMetrics(
            target.generation,
            transition.snapshot_seconds,
            transition.teardown_seconds,
            transition.compile_seconds,
            world_seconds,
            activation_seconds,
            recovery_seconds,
            time.perf_counter() - transition.transition_started,
            detection_seconds=transition.detection_seconds,
            configuration_planning_seconds=transition.configuration_planning_seconds,
            state_planning_seconds=self.last_recovery_report.planning_seconds,
            state_transfer_seconds=self.last_recovery_report.transfer_seconds,
            straggler_round_seconds=self.last_recovery_report.straggler_round_seconds,
            source_scheduling_error_bytes=(self.last_recovery_report.source_scheduling_error_bytes),
        )
    )
    self._prepared_transition = None
    return True


def _activate_latest_generation(self: OobleckParallelContext) -> None:
    """Compatibility path that performs both generation phases synchronously."""

    while self._pending_plan is not None or self._prepared_transition is not None:
        if self._prepared_transition is None or self._pending_plan is not None:
            _prepare_latest_generation(self)
        if self._prepared_transition is not None:
            _activate_prepared_generation(self)


def _apply_membership(self: OobleckParallelContext, snapshot: MembershipSnapshot) -> bool:
    """Validate and plan from one complete membership snapshot.

    Stale snapshots are ignored, while every node must preserve the fixed per-node
    tensor-parallel width. The lowest stable node ID supplies rendezvous, the owner
    plan recomputes topology from the entire membership rather than an incremental
    failure, and detection/configuration timings are retained for acceptance metrics.
    Announcing the plan only queues it; transaction commit and activation barriers
    decide when it can replace the running generation.
    """

    newest_generation = max(
        self.generation,
        self._pending_plan.generation if self._pending_plan is not None else -1,
        (
            self._prepared_transition.target.generation
            if self._prepared_transition is not None
            else -1
        ),
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
    """Require future steps to wait for master-published generation activation."""

    self._control_plane_managed = True
    self._control_active_generation = self.generation


def _prepare_generation(self: OobleckParallelContext) -> None:
    """Run snapshot, teardown, and compile after local preparation consensus."""

    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before preparation")
    if not _prepare_latest_generation(self):
        raise RuntimeError("no newer generation is available to prepare")


def _activate_generation(self: OobleckParallelContext) -> None:
    """Build WORLD, materialize, and recover after rendezvous publication."""

    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before activation")
    if not _activate_prepared_generation(self):
        raise RuntimeError("prepared generation was superseded during activation")


def _prepared_execution_plan(self: OobleckParallelContext) -> Any:
    """Expose the target plan whose checksum local workers must acknowledge."""

    transition = self._prepared_transition
    return self.execution_plan if transition is None else transition.target


def _generation_transition_pending(self: OobleckParallelContext) -> bool:
    """Report whether transactional commit is blocked on generation activation."""

    return self._control_plane_managed and (
        self._pending_plan is not None
        or self._prepared_transition is not None
        or self._control_active_generation != self.generation
    )


def _wait_for_generation_barrier(self: OobleckParallelContext) -> None:
    """Block the training thread until activation or the rendezvous timeout."""

    deadline = time.monotonic() + self.config.rendezvous_timeout_s
    with self._generation_condition:
        while _generation_transition_pending(self):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for generation_active")
            self._generation_condition.wait(timeout=remaining)


def _mark_generation_active(self: OobleckParallelContext, generation: int) -> None:
    """Publish matching master activation to every waiting local step."""

    if generation != self.generation:
        raise RuntimeError(
            f"cannot activate generation {generation}; prepared generation is {self.generation}"
        )
    self._control_active_generation = generation
    with self._generation_condition:
        self._generation_condition.notify_all()


def _step_with_control_barrier(self: OobleckParallelContext, *args: Any, **kwargs: Any) -> Any:
    """Prevent a new transaction attempt from entering an inactive generation."""

    if _generation_transition_pending(self):
        _wait_for_generation_barrier(self)
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
    """Retire synchronization topology, partition state, and the entire WORLD."""

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
OobleckParallelContext.activate_generation = _activate_generation
OobleckParallelContext.prepared_execution_plan = property(_prepared_execution_plan)
OobleckParallelContext._generation_transition_pending = _generation_transition_pending
OobleckParallelContext._wait_for_generation_barrier = _wait_for_generation_barrier
OobleckParallelContext.mark_generation_active = _mark_generation_active
OobleckParallelContext.step = _step_with_control_barrier
OobleckParallelContext.recover_from_survivors = _recover_from_survivors
OobleckParallelContext.close = _close_with_topology
