"""Public runtime with committed-state generation replacement."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from oobleck.distributed import destroy_process_group_universe
from oobleck.elastic.membership import MembershipSnapshot, is_pure_addition
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
    removed_members: tuple[tuple[str, str], ...] = ()
    added_members: tuple[tuple[str, str], ...] = ()
    graceful_cutover: bool = False
    cutover_committed_step: int | None = None


@dataclass(slots=True)
class _PreparedGenerationTransition:
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
    self._deferred_addition_plan = None
    self._step_in_progress = False
    self._transition_metadata: dict[
        int,
        tuple[
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
            bool,
            int | None,
        ],
    ] = {}
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


def _record_superseded(
    self: OobleckParallelContext,
    transition: _PreparedGenerationTransition,
    *,
    world_seconds: float = 0.0,
    activation_seconds: float = 0.0,
    recovery_seconds: float = 0.0,
) -> None:
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
            removed_members=_transition_fields(self, transition.target.generation)[0],
            added_members=_transition_fields(self, transition.target.generation)[1],
            graceful_cutover=_transition_fields(self, transition.target.generation)[2],
            cutover_committed_step=_transition_fields(self, transition.target.generation)[3],
        )
    )


def _transition_fields(
    self: OobleckParallelContext, generation: int
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    bool,
    int | None,
]:
    return self._transition_metadata.get(generation, ((), (), False, None))


def _prepare_latest_generation(self: OobleckParallelContext) -> bool:
    """Retire WORLD and compile the newest ownership without creating a new WORLD."""

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
    """Create WORLD, activate ownership, and recover after rendezvous publication."""

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
            removed_members=_transition_fields(self, target.generation)[0],
            added_members=_transition_fields(self, target.generation)[1],
            graceful_cutover=_transition_fields(self, target.generation)[2],
            cutover_committed_step=_transition_fields(self, target.generation)[3],
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


def _member_ids(nodes: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((node.agent_id, node.incarnation_id) for node in nodes)


def _apply_membership(self: OobleckParallelContext, snapshot: MembershipSnapshot) -> bool:
    """Convert one complete snapshot into a hard or graceful pending generation."""

    newest_generation = max(
        self.generation,
        self._pending_plan.generation if self._pending_plan is not None else -1,
        (
            self._deferred_addition_plan.generation
            if self._deferred_addition_plan is not None
            else -1
        ),
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
    if not snapshot.nodes:
        raise ValueError("membership must retain at least one node")
    previous = snapshot.previous_execution_plan
    if previous is not None:
        if (
            self.execution_plan.generation == previous.generation
            and self.execution_plan.plan_checksum != previous.plan_checksum
        ):
            raise ValueError("membership previous plan disagrees with the active worker plan")
        self.owner_plan._last_execution_plan = previous
    coordinator = min(snapshot.nodes, key=lambda node: node.agent_id)
    self.owner_plan.set_rendezvous_address(coordinator.addresses[0])
    planning_started = time.perf_counter()
    self.owner_plan.set_membership(
        tuple(node.agent_id for node in snapshot.nodes), snapshot.generation
    )
    execution_plan = self.owner_plan.build_execution_plan()
    planning_seconds = time.perf_counter() - planning_started
    detection_seconds = snapshot.detection_seconds
    self._generation_control_metrics[snapshot.generation] = (
        detection_seconds,
        planning_seconds,
    )
    graceful = is_pure_addition(snapshot, self.execution_plan)
    with self._commit_lock:
        deferred = graceful and self._step_in_progress
        self._transition_metadata[snapshot.generation] = (
            _member_ids(snapshot.removed_nodes),
            _member_ids(snapshot.added_nodes),
            graceful,
            None if deferred else self.committed_step,
        )
        if deferred:
            self._deferred_addition_plan = execution_plan
        else:
            self._deferred_addition_plan = None
            if (
                self._pending_plan is None
                or execution_plan.generation > self._pending_plan.generation
            ):
                self._pending_plan = execution_plan
    with self._generation_condition:
        self._generation_condition.notify_all()
    return True


def _enable_control_plane_barrier(self: OobleckParallelContext) -> None:
    self._control_plane_managed = True
    self._control_active_generation = self.generation


def _prepare_generation(self: OobleckParallelContext) -> None:
    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before preparation")
    if not _prepare_latest_generation(self):
        raise RuntimeError("no newer generation is available to prepare")


def _activate_generation(self: OobleckParallelContext) -> None:
    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before activation")
    if not _activate_prepared_generation(self):
        raise RuntimeError("prepared generation was superseded during activation")


def _prepared_execution_plan(self: OobleckParallelContext) -> Any:
    transition = self._prepared_transition
    return self.execution_plan if transition is None else transition.target


def _generation_transition_pending(self: OobleckParallelContext) -> bool:
    return self._control_plane_managed and (
        self._pending_plan is not None
        or self._prepared_transition is not None
        or self._control_active_generation != self.generation
    )


def _wait_for_generation_barrier(self: OobleckParallelContext) -> None:
    deadline = time.monotonic() + self.config.rendezvous_timeout_s
    with self._generation_condition:
        while _generation_transition_pending(self):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for generation_active")
            self._generation_condition.wait(timeout=remaining)


def _mark_generation_active(self: OobleckParallelContext, generation: int) -> None:
    if generation != self.generation:
        raise RuntimeError(
            f"cannot activate generation {generation}; prepared generation is {self.generation}"
        )
    self._control_active_generation = generation
    with self._generation_condition:
        self._generation_condition.notify_all()


def _wait_until_generation_preparable(self: OobleckParallelContext, generation: int) -> bool:
    """Wait for a graceful step boundary; return false when superseded."""

    deadline = time.monotonic() + self.config.rendezvous_timeout_s
    with self._generation_condition:
        while True:
            prepared_generation = (
                self._prepared_transition.target.generation
                if self._prepared_transition is not None
                else -1
            )
            pending_generation = (
                self._pending_plan.generation if self._pending_plan is not None else -1
            )
            deferred_generation = (
                self._deferred_addition_plan.generation
                if self._deferred_addition_plan is not None
                else -1
            )
            newest = max(prepared_generation, pending_generation, deferred_generation)
            if newest > generation:
                return False
            if pending_generation == generation or prepared_generation == generation:
                return True
            if newest < generation:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for graceful addition boundary")
            self._generation_condition.wait(timeout=remaining)


def _step_with_control_barrier(self: OobleckParallelContext, *args: Any, **kwargs: Any) -> Any:
    while True:
        if _generation_transition_pending(self):
            _wait_for_generation_barrier(self)
        with self._commit_lock:
            if _generation_transition_pending(self):
                continue
            self._step_in_progress = True
            break
    try:
        return _base_step(self, *args, **kwargs)
    finally:
        with self._commit_lock:
            self._step_in_progress = False
            target = self._deferred_addition_plan
            self._deferred_addition_plan = None
            if target is not None and (
                self._pending_plan is None or target.generation > self._pending_plan.generation
            ):
                self._pending_plan = target
                removed, added, graceful, _ = self._transition_metadata[target.generation]
                self._transition_metadata[target.generation] = (
                    removed,
                    added,
                    graceful,
                    self.committed_step,
                )
        with self._generation_condition:
            self._generation_condition.notify_all()


def _recover_from_survivors(self: OobleckParallelContext) -> None:
    """Bootstrap an added worker that owns no pre-generation state."""

    if not self._needs_survivor_recovery:
        raise RuntimeError("materialize with recover_from_survivors=True for an added worker")
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
OobleckParallelContext.activate_generation = _activate_generation
OobleckParallelContext.prepared_execution_plan = property(_prepared_execution_plan)
OobleckParallelContext._generation_transition_pending = _generation_transition_pending
OobleckParallelContext._wait_for_generation_barrier = _wait_for_generation_barrier
OobleckParallelContext.wait_until_generation_preparable = _wait_until_generation_preparable
OobleckParallelContext.mark_generation_active = _mark_generation_active
OobleckParallelContext.step = _step_with_control_barrier
OobleckParallelContext.recover_from_survivors = _recover_from_survivors
OobleckParallelContext.close = _close_with_topology
