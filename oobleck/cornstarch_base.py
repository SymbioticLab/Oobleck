"""Lazy boundary around Cornstarch's compile/activate partition lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from oobleck.state import LogicalStateEntry, StateManifest
from oobleck.types import OobleckExecutionPlan, PipelineStageSpec


def _local_manifest(model: torch.nn.Module, rank: int, committed_step: int = 0) -> StateManifest:
    entries = []
    for kind, values in (
        ("parameter", model.named_parameters(recurse=True)),
        ("buffer", model.named_buffers(recurse=True)),
    ):
        for name, value in values:
            entries.append(
                LogicalStateEntry(
                    logical_key=name,
                    global_shape=tuple(value.shape),
                    local_shape=tuple(value.shape),
                    dtype=str(value.dtype),
                    state_kind=kind,
                    placements=("replicate",),
                    tp_lane=0,
                    owner_rank=rank,
                    version=committed_step,
                )
            )
    return StateManifest(rank, tuple(entries), committed_step)


@dataclass(frozen=True, slots=True)
class CompiledLocalPartition:
    """Process-group-independent rank-local ownership description."""

    root_model: torch.nn.Module
    rank: int
    world_size: int
    stage_spec: PipelineStageSpec
    manifest: StateManifest
    external_compiled: Any = None

    def activate(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        *,
        mesh: Any = None,
    ) -> "ActivatedPartition":
        if self.external_compiled is not None:
            external = self.external_compiled.activate(device=device, dtype=dtype, mesh=mesh)
            model = getattr(external, "model", self.root_model)
            return ActivatedPartition(self, model, external)
        target = torch.device(device)
        parameters = tuple(self.root_model.parameters())
        if parameters and any(parameter.is_meta for parameter in parameters):
            self.root_model.to_empty(device=target)
            initializer = getattr(self.root_model, "initialize_checkpoint", None)
            if callable(initializer):
                initializer()
        else:
            self.root_model.to(device=target, dtype=dtype)
        return ActivatedPartition(self, self.root_model)


class ActivatedPartition:
    def __init__(
        self,
        compiled: CompiledLocalPartition,
        model: torch.nn.Module,
        external_context: Any = None,
    ) -> None:
        self.compiled = compiled
        self.model = model
        self.external_context = external_context
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        if self.external_context is not None:
            closer = getattr(self.external_context, "close", None)
            if callable(closer):
                closer()
        self.external_context = None
        self._closed = True


def compile_local_partition(
    model: torch.nn.Module,
    parallel_config: object,
    execution_plan: OobleckExecutionPlan,
    rank: int,
    *,
    cornstarch_plan: object | None = None,
) -> CompiledLocalPartition:
    """Compile ownership without touching ``torch.distributed`` or real storage."""

    stage = execution_plan.rank_local_stage(rank)
    external = None
    if cornstarch_plan is not None:
        compiler = getattr(cornstarch_plan, "compile", None)
        if compiler is None:
            raise RuntimeError(
                "The installed Cornstarch does not expose compile(). Install the "
                "pinned refactor-oobleck commit rather than a moving branch."
            )
        external = compiler(
            world_size=sum(len(ranks) for _, ranks in execution_plan.rank_map),
            rank=rank,
            stage_overrides=(stage,),
        )
    return CompiledLocalPartition(
        model,
        rank,
        sum(len(ranks) for _, ranks in execution_plan.rank_map),
        stage,
        _local_manifest(model, rank),
        external,
    )
