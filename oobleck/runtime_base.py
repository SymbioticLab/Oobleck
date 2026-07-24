"""Public plan/context API and logical-step transaction implementation."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from oobleck.cornstarch import ActivatedPartition, CompiledLocalPartition, compile_local_partition
from oobleck.data import (
    OobleckBatch,
    OobleckBatchSampler,
    PreparedDataLoader,
    logical_seed,
    prepare_dataloader,
)
from oobleck.distributed import destroy_process_group_universe, initialize_process_group
from oobleck.planning import (
    PipelineTemplate,
    RecoveryUnavailable,
    compose_templates,
    reconfigure_pipelines,
)
from oobleck.types import (
    OobleckConfig,
    OobleckExecutionPlan,
    RuntimeCompatibility,
    stable_rank_map,
)


@dataclass(frozen=True, slots=True)
class OobleckStepResult:
    """Outcome of one logical batch, including retries across generations."""

    committed: bool
    committed_step: int
    generation: int
    attempts: int
    loss: float | None = None
    output: Any = None


@dataclass(slots=True)
class OobleckPreparedContext:
    """Rank-local ownership compiled before the distributed WORLD exists."""

    owner_plan: "OobleckParallelizationPlan"
    execution_plan: OobleckExecutionPlan
    compiled: CompiledLocalPartition
    device: torch.device
    dtype: torch.dtype | None
    recover_from_survivors: bool = False
    _activated: bool = False

    def activate(self) -> "OobleckParallelContext":
        """Initialize WORLD, materialize local state, and consume this preparation once."""

        if self._activated:
            raise RuntimeError("prepared context has already been activated")
        if self.owner_plan.world_initializer is not None:
            self.owner_plan.world_initializer(self.execution_plan)
        partition = self.compiled.activate(self.device, self.dtype)
        self._activated = True
        return OobleckParallelContext(
            config=self.owner_plan.config,
            owner_plan=self.owner_plan,
            execution_plan=self.execution_plan,
            compiled=self.compiled,
            partition=partition,
            device=self.device,
            dtype=self.dtype,
            needs_survivor_recovery=self.recover_from_survivors,
        )


class OobleckParallelizationPlan:
    """Prepare rank-local ownership before creating the distributed world."""

    def __init__(
        self,
        config: OobleckConfig,
        *,
        templates: Sequence[PipelineTemplate] = (),
        node_ids: Sequence[str] = (),
        rank: int = 0,
        cornstarch_plan: object | None = None,
        world_initializer: Callable[[OobleckExecutionPlan], None] | None = None,
        compatibility: RuntimeCompatibility | None = None,
    ) -> None:
        """Record planning inputs without touching the distributed data plane."""

        self.config = config
        self.templates = tuple(templates)
        self.node_ids = tuple(node_ids)
        self.rank = rank
        self.cornstarch_plan = cornstarch_plan
        self.world_initializer = world_initializer
        self.compatibility = compatibility
        self.model: torch.nn.Module | None = None
        self.parallel_config: object | None = None
        self._generation = 0
        self._last_execution_plan: OobleckExecutionPlan | None = None
        self.last_reconfiguration: object | None = None
        self._local_node_id: str | None = None
        self._tp_lane: int | None = None
        self._target_rank = rank
        self._rendezvous_address: str | None = None

    def parallelize(self, model: torch.nn.Module, parallel_config: object) -> None:
        """Register the reusable model blueprint and validate Oobleck-owned dimensions."""

        if self.model is not None:
            raise RuntimeError("parallelize() may be called only once in the first release")
        required = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": None,
            "data_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_parallel_size": 1,
        }
        for name, expected in required.items():
            if not hasattr(parallel_config, name):
                if name == "tensor_parallel_size":
                    raise TypeError("parallel_config must define tensor_parallel_size")
                continue
            value = getattr(parallel_config, name)
            if name == "tensor_parallel_size":
                if not isinstance(value, int) or value < 1:
                    raise ValueError("tensor_parallel_size must be a positive integer")
            elif value != expected:
                raise ValueError(
                    f"Oobleck chooses heterogeneous PP/DP and requires {name}={expected!r}; got {value!r}"
                )
        self.model = model
        self.parallel_config = parallel_config
        if self.cornstarch_plan is not None:
            register = getattr(self.cornstarch_plan, "parallelize", None)
            if callable(register):
                register(model, parallel_config)

    def set_templates(self, templates: Sequence[PipelineTemplate]) -> None:
        """Replace the candidate template catalog used for future generations."""

        if not templates:
            raise ValueError("templates must not be empty")
        self.templates = tuple(templates)

    def set_membership(self, node_ids: Sequence[str], generation: int | None = None) -> None:
        """Install a sorted membership and monotonically advance its generation."""

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        if len(node_ids) > self.config.max_nodes:
            raise ValueError("membership exceeds configured max_nodes")
        selected = tuple(sorted(node_ids))
        changed = selected != tuple(sorted(self.node_ids))
        self.node_ids = selected
        if generation is not None:
            if generation < self._generation:
                raise ValueError("membership generation cannot move backwards")
            self._generation = generation
        elif changed:
            self._generation += 1

    def set_rendezvous_address(self, address: str) -> None:
        """Select the coordinator address used when creating replacement WORLDs."""

        if not address:
            raise ValueError("rendezvous address must not be empty")
        self._rendezvous_address = address
        if self.world_initializer is None:
            self.world_initializer = self.initialize_world

    def initialize_world(self, execution_plan: OobleckExecutionPlan) -> None:
        """Initialize WORLD for a managed generation using its concrete rank map."""

        if self._rendezvous_address is None:
            raise RuntimeError("managed WORLD initialization requires a rendezvous address")
        world_size = sum(len(ranks) for _, ranks in execution_plan.rank_map)
        backend = self.config.distributed_backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        rendezvous_port = self.config.rendezvous_port + execution_plan.generation
        if rendezvous_port > 65535:
            raise ValueError("rendezvous_port plus membership generation exceeds 65535")
        initialize_process_group(
            backend=backend,
            master_address=self._rendezvous_address,
            master_port=rendezvous_port,
            rank=self.rank_for_plan(execution_plan),
            world_size=world_size,
            timeout_s=self.config.rendezvous_timeout_s,
        )

    def _default_template(self) -> PipelineTemplate:
        """Build a local single-stage fallback when no profiled templates were supplied."""

        assert self.model is not None and self.parallel_config is not None
        layer_count = 1
        repeated_layers = getattr(self.model, "_repeated_layers", None)
        if callable(repeated_layers):
            layers = repeated_layers()
            if len(layers):
                layer_count = len(layers)
        for candidate in ("layers", "h", "blocks"):
            value = getattr(self.model, candidate, None)
            if isinstance(value, (torch.nn.ModuleList, list, tuple)) and value:
                layer_count = len(value)
                break
        return PipelineTemplate(
            "local-single-stage",
            ((0, layer_count),),
            int(getattr(self.parallel_config, "tensor_parallel_size")),
            1.0,
            1.0,
        )

    def build_execution_plan(self) -> OobleckExecutionPlan:
        """Compose or reconfigure pipelines and assign deterministic concrete ranks."""

        if self.model is None or self.parallel_config is None:
            raise RuntimeError("call parallelize() before building or materializing a plan")
        tp = int(getattr(self.parallel_config, "tensor_parallel_size"))
        nodes = self.node_ids or (os.environ.get("OOBLECK_NODE_ID", "local-node"),)
        if len(nodes) > self.config.max_nodes:
            raise ValueError("membership exceeds configured max_nodes")
        templates = self.templates or (self._default_template(),)
        if any(item.tensor_parallel_size != tp for item in templates):
            raise ValueError("every template TP width must equal the fixed per-node TP width")
        minimum = min(item.resource_count for item in templates)
        threshold = self.config.fault_tolerance_threshold
        if len(nodes) < minimum * (threshold + 1):
            threshold = 0
        previous_plan = self._last_execution_plan
        if previous_plan is not None and previous_plan.generation < self._generation:
            state_bytes_by_node = {
                node: max(1, instance.template.persistent_memory // len(instance.node_ids))
                for instance in previous_plan.instances
                for node in instance.node_ids
            }
            reconfiguration = reconfigure_pipelines(
                previous_plan.instances,
                nodes,
                templates,
                self.config.global_num_microbatches,
                threshold,
                state_bytes_by_node=state_bytes_by_node,
            )
            instances = reconfiguration.instances
            self.last_reconfiguration = reconfiguration
        else:
            instances = compose_templates(
                templates,
                nodes,
                self.config.global_num_microbatches,
                threshold,
            )
            self.last_reconfiguration = None
        rank_map = stable_rank_map(nodes, tp)
        ranks_by_node = dict(rank_map)
        ranked = tuple(
            replace(
                item,
                ranks=tuple(ranks_by_node[node] for node in item.node_ids),
            )
            for item in instances
        )
        previous = (
            previous_plan.generation
            if previous_plan is not None and previous_plan.generation < self._generation
            else (
                previous_plan.previous_generation
                if previous_plan is not None
                else (self._generation - 1 if self._generation else None)
            )
        )
        result = OobleckExecutionPlan(
            self._generation,
            ranked,
            rank_map,
            previous,
            self.compatibility.digest if self.compatibility is not None else None,
        )
        self._last_execution_plan = result
        if self._local_node_id is None:
            for node_id, rank_group in result.rank_map:
                if self.rank in rank_group:
                    self._local_node_id = node_id
                    self._tp_lane = rank_group.index(self.rank)
                    break
        return result

    def rank_for_plan(self, execution_plan: OobleckExecutionPlan) -> int:
        """Preserve this worker's node and TP-lane identity across rank remapping."""

        if self._local_node_id is None:
            return self.rank
        for node_id, ranks in execution_plan.rank_map:
            if node_id == self._local_node_id:
                lane = self._tp_lane or 0
                if lane >= len(ranks):
                    raise RecoveryUnavailable(f"TP lane {lane} is unavailable for node {node_id!r}")
                return ranks[lane]
        raise RecoveryUnavailable(
            f"local node {self._local_node_id!r} is not part of generation "
            f"{execution_plan.generation}"
        )

    def compile(self, execution_plan: OobleckExecutionPlan | None = None) -> CompiledLocalPartition:
        """Compile rank-local ownership before WORLD, after compatibility validation."""

        if self.model is None or self.parallel_config is None:
            raise RuntimeError("call parallelize() before compile()")
        plan = execution_plan or self.build_execution_plan()
        expected_compatibility = (
            self.compatibility.digest if self.compatibility is not None else None
        )
        if plan.compatibility_digest != expected_compatibility:
            raise ValueError(
                "execution plan compatibility does not match this worker; "
                f"plan={plan.compatibility_digest}, local={expected_compatibility}"
            )
        target_rank = self.rank_for_plan(plan)
        self._target_rank = target_rank
        return compile_local_partition(
            self.model,
            self.parallel_config,
            plan,
            target_rank,
            cornstarch_plan=self.cornstarch_plan,
        )

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        *,
        recover_from_survivors: bool = False,
    ) -> "OobleckParallelContext":
        """Compatibility wrapper that prepares and immediately activates a partition."""

        return self.prepare(
            device,
            dtype,
            recover_from_survivors=recover_from_survivors,
        ).activate()

    def prepare(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        *,
        recover_from_survivors: bool = False,
    ) -> OobleckPreparedContext:
        """Compile local ownership without initializing WORLD or allocating state."""

        execution_plan = self.build_execution_plan()
        compiled = self.compile(execution_plan)
        return OobleckPreparedContext(
            owner_plan=self,
            execution_plan=execution_plan,
            compiled=compiled,
            device=torch.device(device),
            dtype=dtype,
            recover_from_survivors=recover_from_survivors,
        )


class OobleckParallelContext:
    """Active generation plus transactional training and reconfiguration state."""

    def __init__(
        self,
        *,
        config: OobleckConfig,
        owner_plan: OobleckParallelizationPlan,
        execution_plan: OobleckExecutionPlan,
        compiled: CompiledLocalPartition,
        partition: ActivatedPartition,
        device: torch.device,
        dtype: torch.dtype | None,
        needs_survivor_recovery: bool = False,
    ) -> None:
        """Bind an activated local partition to one immutable execution plan."""

        self.config = config
        self.owner_plan = owner_plan
        self.execution_plan = execution_plan
        self.compiled = compiled
        self.partition = partition
        self.device = device
        self.dtype = dtype
        self.committed_step = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.scaler: Any = None
        self._pending_plan: OobleckExecutionPlan | None = None
        self._loaders: list[PreparedDataLoader] = []
        self._closed = False
        self._needs_survivor_recovery = needs_survivor_recovery
        self._commit_lock = threading.RLock()

    @property
    def model(self) -> torch.nn.Module:
        """Return the currently active rank-local model view."""

        return self.partition.model

    @property
    def generation(self) -> int:
        """Return the active membership generation."""

        return self.execution_plan.generation

    def create_batch_sampler(
        self,
        dataset: object,
        *,
        shuffle: bool,
        drop_last: bool = True,
    ) -> OobleckBatchSampler:
        """Create a commit-aware sampler using this generation's allocation."""

        return OobleckBatchSampler(
            dataset,
            global_batch_size=self.config.global_batch_size,
            microbatch_size=self.config.microbatch_size,
            instances=self.execution_plan.instances,
            seed=self.config.seed,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def prepare_dataloader(self, dataloader: DataLoader) -> PreparedDataLoader:
        """Attach a validated loader so reconfiguration can invalidate its prefetch."""

        loader = prepare_dataloader(dataloader)
        self._loaders.append(loader)
        return loader

    def configure_optimization(
        self,
        *,
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
        scaler: Any = None,
    ) -> None:
        """Construct optimization state only after local parameters are materialized."""

        if self.optimizer is not None:
            raise RuntimeError("optimization is already configured")
        self.optimizer = optimizer_factory(self.model.parameters())
        self.scheduler = scheduler_factory(self.optimizer) if scheduler_factory else None
        self.scaler = scaler

    def announce_generation(self, plan: OobleckExecutionPlan) -> bool:
        """Queue only the newest future plan under the optimizer commit lock."""

        with self._commit_lock:
            if plan.generation <= self.generation:
                return False
            if self._pending_plan is None or plan.generation > self._pending_plan.generation:
                self._pending_plan = plan
            return True

    def _activate_latest_generation(self) -> None:
        """Retire WORLD completely and activate the newest queued generation."""

        while self._pending_plan is not None:
            target = self._pending_plan
            self._pending_plan = None
            for loader in self._loaders:
                loader.invalidate_prefetch()
            self.partition.close()
            try:
                import torch.distributed as dist

                initialized = dist.is_initialized()
            except Exception:
                initialized = False
            if initialized:
                destroy_process_group_universe()
            compiled = self.owner_plan.compile(target)
            if self.owner_plan.world_initializer is not None:
                self.owner_plan.world_initializer(target)
            self.compiled = compiled
            self.partition = compiled.activate(self.device, self.dtype)
            self.execution_plan = target

    def _call_model(self, microbatch: Any) -> Any:
        """Dispatch mapping batches as keyword arguments and other batches positionally."""

        if isinstance(microbatch, Mapping):
            return self.model(**microbatch)
        return self.model(microbatch)

    def _local_microbatches(self, batch: OobleckBatch) -> tuple[tuple[int, Any], ...]:
        """Select this pipeline's global microbatch IDs from the logical batch."""

        pipeline_id = self.execution_plan.rank_local_stage(self.owner_plan.rank).pipeline_id
        cursor = 0
        selected: tuple[int, ...] | None = None
        for instance in self.execution_plan.instances:
            identifiers = tuple(range(cursor, cursor + instance.microbatches))
            if instance.instance_id == pipeline_id:
                selected = identifiers
            cursor += instance.microbatches
        if cursor != len(batch.microbatches):
            raise RuntimeError(
                f"logical batch has {len(batch.microbatches)} microbatches but "
                f"generation {self.generation} allocates {cursor}"
            )
        if selected is None:
            raise RuntimeError(f"local pipeline {pipeline_id!r} has no batch allocation")
        return tuple((index, batch.microbatches[index]) for index in selected)

    def _execute_attempt(
        self,
        batch: OobleckBatch,
        execution: Any,
        output_reference: Any,
        criterion: Callable[..., torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, Any]:
        """Run one forward/backward attempt with replay-stable stochastic seeds."""

        assert self.optimizer is not None
        local_microbatches = self._local_microbatches(batch)
        if hasattr(execution, "step") and callable(execution.step):
            result = execution.step(
                [microbatch for _, microbatch in local_microbatches],
                criterion,
                self.optimizer,
                return_loss=True,
            )
            loss = result.get("loss") if isinstance(result, dict) else None
            return loss, result
        total_loss: torch.Tensor | None = None
        last_output: Any = None
        count = max(1, len(local_microbatches))
        for microbatch_id, microbatch in local_microbatches:
            seed = logical_seed(
                self.config.seed,
                batch.descriptor.epoch,
                self.committed_step,
                microbatch_id,
            )
            devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if execution is None:
                    last_output = self._call_model(microbatch)
                elif callable(execution):
                    last_output = execution(microbatch)
                elif hasattr(execution, "execute"):
                    last_output = execution.execute(microbatch)
                else:
                    raise TypeError("execution plan must be callable or expose execute()/step()")
                loss = criterion(last_output, microbatch) if criterion else last_output
                if not isinstance(loss, torch.Tensor):
                    raise TypeError("criterion/execution must produce a torch.Tensor loss")
                (loss / count).backward()
                total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
        return (None if total_loss is None else total_loss / count), last_output

    def step(
        self,
        batch: OobleckBatch,
        execution_plan: Any = None,
        output: Any = None,
        criterion: Callable[..., torch.Tensor] | None = None,
    ) -> OobleckStepResult:
        """Execute and atomically commit a logical batch, replaying on generation change.

        Optimizer, scheduler, scaler, step, and sampler cursors advance together
        only after gradient synchronization and a final membership check.
        """

        if self._closed:
            raise RuntimeError("context is closed")
        if self.optimizer is None:
            raise RuntimeError("call configure_optimization() before step()")

        transition_pending = getattr(self, "_generation_transition_pending", None)
        wait_for_generation = getattr(self, "_wait_for_generation_barrier", None)

        def managed_transition_pending() -> bool:
            """Consult the optional control-plane barrier installed by runtime.py."""

            return callable(transition_pending) and bool(transition_pending())

        def wait_if_managed() -> None:
            """Block commits while the master coordinates a two-phase transition."""

            if managed_transition_pending():
                if not callable(wait_for_generation):
                    raise RuntimeError("managed generation has no control-plane waiter")
                wait_for_generation()

        attempts = 0
        while True:
            wait_if_managed()
            attempts += 1
            if self._pending_plan is not None:
                self._activate_latest_generation()
            attempt_generation = self.generation
            self.optimizer.zero_grad(set_to_none=True)
            try:
                loss, result = self._execute_attempt(batch, execution_plan, output, criterion)
                sync = getattr(self.partition.external_context, "sync_gradients", None)
                if callable(sync):
                    sync()
                heterogeneous_sync = getattr(self, "_heterogeneous_gradient_sync", None)
                if heterogeneous_sync is not None:
                    heterogeneous_sync.sync()
            except Exception:
                if self.generation == attempt_generation and not managed_transition_pending():
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                wait_if_managed()
                continue

            retry = False
            with self._commit_lock:
                if (
                    self.generation != attempt_generation
                    or self._pending_plan is not None
                    or managed_transition_pending()
                ):
                    retry = True
                else:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.committed_step += 1
                    for loader in self._loaders:
                        if (
                            loader.sampler.descriptor_at(loader.sampler.committed_cursor)
                            == batch.descriptor
                        ):
                            loader.sampler.commit(batch.descriptor)
                            break
            if retry:
                self.optimizer.zero_grad(set_to_none=True)
                if managed_transition_pending():
                    wait_if_managed()
                elif self._pending_plan is not None:
                    self._activate_latest_generation()
                continue
            return OobleckStepResult(
                True,
                self.committed_step,
                self.generation,
                attempts,
                None if loss is None else float(loss.cpu()),
                result,
            )

    def close(self) -> None:
        """Idempotently retire the local partition and attached loader references."""

        if self._closed:
            return
        self.partition.close()
        self._loaders.clear()
        self._closed = True

    def __enter__(self) -> "OobleckParallelContext":
        """Return the active context for managed use."""

        return self

    def __exit__(self, *exc_info: object) -> None:
        """Retire resources regardless of how the managed block exits."""

        self.close()
