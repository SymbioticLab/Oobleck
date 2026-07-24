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
    """Timing and membership facts recorded for one transition attempt.

    Superseded attempts remain visible for diagnosis. ``graceful_cutover``
    identifies pure additions, while ``cutover_committed_step`` records the
    committed boundary after which the new cohort was allowed to prepare.
    """

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
    """Resources and timings retained between prepare and activation barriers."""

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
    """Extend a context with recovery, transition, and cutover bookkeeping.

    Optimizer factories are retained for rebuilding ownership, while condition
    and commit state coordinate external generation barriers with training steps.
    Separate pending, prepared, and deferred-addition slots make the transition
    lifecycle explicit. Existing owners enable gradient synchronization at once;
    added workers postpone it until survivor state has been restored.
    """
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
    """Configure optimization and retain factories needed after repartitioning."""
    _base_configure_optimization(
        self,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scaler=scaler,
    )
    self._optimizer_factory = optimizer_factory
    self._scheduler_factory = scheduler_factory


def _world_initialized() -> bool:
    """Return whether torch.distributed has a live default process group."""
    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except (ImportError, RuntimeError):
        return False


def _retire_world(self: OobleckParallelContext) -> None:
    """Close topology resources before destroying the current process-group universe."""
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
    """Append metrics once for a proposal superseded during preparation.

    A proposal can become obsolete before WORLD creation, during activation, or
    after recovery. Optional phase durations describe how far it progressed. Any
    completed recovery report contributes redistribution timings, and transition
    metadata preserves the coalesced removed/added identities and cutover boundary.
    The per-transition guard prevents duplicate history entries across cleanup paths.
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
    """Return removed/added identities and graceful-boundary metadata."""
    return self._transition_metadata.get(generation, ((), (), False, None))


def _prepare_latest_generation(self: OobleckParallelContext) -> bool:
    """Snapshot committed state, retire WORLD, and compile the newest proposal.

    The first proposal captures state and tears down the old world exactly once.
    If newer complete membership arrives while compilation is underway, the
    obsolete attempt is recorded and compilation restarts from the same committed
    snapshot. No process groups for the target generation are created until the
    external prepared/rendezvous barrier succeeds.
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
    """Create WORLD, activate ownership, and restore state after rendezvous.

    Activation installs the compiled partition and execution plan, reconfigures
    data loaders, restores the committed snapshot, then enables heterogeneous
    gradient synchronization. A superseding generation aborts this attempt and
    reuses the captured snapshot; only the newest successful activation is added
    to recovery history as the live generation.
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
    """Project node identities into stable-ID/incarnation metric tuples."""
    return tuple((node.agent_id, node.incarnation_id) for node in nodes)


def _apply_membership(self: OobleckParallelContext, snapshot: MembershipSnapshot) -> bool:
    """Convert a complete membership snapshot into a pending execution plan.

    The published previous plan is validated against local ownership before the
    new complete cohort is planned. A proposal with additions and no removals is
    graceful: if a step is running, it is deferred until that step commits, so
    its gradients are not replayed. Any removal—including an old incarnation
    removed when the same stable agent ID restarts—is a hard transition and is
    made pending immediately. Newer proposals supersede older deferred plans.
    """

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
    """Require future generations to pass external prepare and active barriers."""
    self._control_plane_managed = True
    self._control_active_generation = self.generation


def _prepare_generation(self: OobleckParallelContext) -> None:
    """Compile the pending generation after the control plane admits preparation."""
    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before preparation")
    if not _prepare_latest_generation(self):
        raise RuntimeError("no newer generation is available to prepare")


def _activate_generation(self: OobleckParallelContext) -> None:
    """Activate the prepared generation after the global rendezvous is published."""
    if not self._control_plane_managed:
        raise RuntimeError("enable the control-plane barrier before activation")
    if not _activate_prepared_generation(self):
        raise RuntimeError("prepared generation was superseded during activation")


def _prepared_execution_plan(self: OobleckParallelContext) -> Any:
    """Expose the candidate plan used in prepared consensus, or the active plan."""
    transition = self._prepared_transition
    return self.execution_plan if transition is None else transition.target


def _generation_transition_pending(self: OobleckParallelContext) -> bool:
    """Return whether managed execution must remain behind a generation barrier."""
    return self._control_plane_managed and (
        self._pending_plan is not None
        or self._prepared_transition is not None
        or self._control_active_generation != self.generation
    )


def _wait_for_generation_barrier(self: OobleckParallelContext) -> None:
    """Block step admission until the prepared generation is globally active."""
    deadline = time.monotonic() + self.config.rendezvous_timeout_s
    with self._generation_condition:
        while _generation_transition_pending(self):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for generation_active")
            self._generation_condition.wait(timeout=remaining)


def _mark_generation_active(self: OobleckParallelContext, generation: int) -> None:
    """Release waiting steps after validating the control plane's active generation."""
    if generation != self.generation:
        raise RuntimeError(
            f"cannot activate generation {generation}; prepared generation is {self.generation}"
        )
    self._control_active_generation = generation
    with self._generation_condition:
        self._generation_condition.notify_all()


def _wait_until_generation_preparable(self: OobleckParallelContext, generation: int) -> bool:
    """Wait until ``generation`` can be prepared without cutting through a step.

    A pure-addition plan stays deferred while the current step commits. The step
    epilogue promotes it to pending and wakes this waiter. The method returns
    ``False`` instead of preparing when a newer generation supersedes the target
    or the target is no longer known.
    """

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
    """Run one optimizer step without crossing a generation transition.

    Step admission waits for every managed generation to become active, then the
    commit lock atomically marks the step in progress. Hard transitions can stop
    subsequent work immediately. A pure addition received mid-step is promoted
    only in the ``finally`` block after the base step has committed; that boundary
    is recorded in transition metrics and no partial gradient replay is needed.
    """
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
    """Close gradient topology, context resources, and the distributed universe."""
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
