"""Lazy boundary around Cornstarch's compile/activate partition lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from oobleck.meshes import create_heterogeneous_pipeline_meshes
from oobleck.state import LogicalStateEntry, StateManifest
from oobleck.types import OobleckExecutionPlan, PipelineStageSpec


def _fallback_manifest(model: torch.nn.Module, rank: int, committed_step: int = 0) -> StateManifest:
    entries = []
    for kind, values in (
        ("parameter", model.named_parameters(recurse=True)),
        ("buffer", model.named_buffers(recurse=True)),
    ):
        for name, value in values:
            entries.append(
                LogicalStateEntry(
                    name,
                    tuple(value.shape),
                    tuple(value.shape),
                    str(value.dtype),
                    kind,
                    ("replicate",),
                    0,
                    rank,
                    committed_step,
                )
            )
    return StateManifest(rank, tuple(entries), committed_step)


def _external_manifest(value: Any, rank: int, committed_step: int = 0) -> StateManifest:
    entries = []
    for item in value.entries:
        entries.append(
            LogicalStateEntry(
                item.logical_key,
                tuple(item.global_shape),
                tuple(item.local_shape),
                str(item.dtype),
                item.state_kind,
                tuple(item.placements),
                int(item.tp_lane),
                rank,
                committed_step,
                None if item.global_layer_id is None else str(item.global_layer_id),
                item.shared_state_id,
            )
        )
    return StateManifest(rank, tuple(entries), committed_step)


@dataclass(frozen=True, slots=True)
class CompiledLocalPartition:
    root_model: torch.nn.Module
    rank: int
    world_size: int
    stage_spec: PipelineStageSpec
    manifest: StateManifest
    execution_plan: OobleckExecutionPlan
    external_compiled: Any = None

    def activate(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        *,
        mesh: Any = None,
    ) -> "ActivatedPartition":
        if self.external_compiled is not None:
            all_meshes = None
            if mesh is None and self.world_size > 1:
                all_meshes, mesh = create_heterogeneous_pipeline_meshes(
                    self.execution_plan,
                    device_type=torch.device(device).type,
                )
            external = self.external_compiled.activate(device=device, dtype=dtype, mesh=mesh)
            model = getattr(external, "model", self.root_model)
            manifest_value = getattr(external, "local_state_manifest", None)
            manifest = (
                self.manifest
                if manifest_value is None
                else _external_manifest(manifest_value, self.rank)
            )
            return ActivatedPartition(self, model, external, manifest, all_meshes)
        target = torch.device(device)
        parameters = tuple(self.root_model.parameters())
        if parameters and any(parameter.is_meta for parameter in parameters):
            self.root_model.to_empty(device=target)
            initializer = getattr(self.root_model, "initialize_checkpoint", None)
            if callable(initializer):
                initializer()
        else:
            self.root_model.to(device=target, dtype=dtype)
        return ActivatedPartition(self, self.root_model, manifest=self.manifest)


class ActivatedPartition:
    def __init__(
        self,
        compiled: CompiledLocalPartition,
        model: torch.nn.Module,
        external_context: Any = None,
        manifest: StateManifest | None = None,
        all_meshes: dict[str, Any] | None = None,
    ) -> None:
        self.compiled = compiled
        self.model = model
        self.external_context = external_context
        self.manifest = manifest or compiled.manifest
        self.all_meshes = all_meshes or {}
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
        self.all_meshes.clear()
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
    manifest = _fallback_manifest(model, rank)
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
        if hasattr(external, "local_state_manifest"):
            manifest = _external_manifest(external.local_state_manifest, rank)
    return CompiledLocalPartition(
        model,
        rank,
        sum(len(ranks) for _, ranks in execution_plan.rank_map),
        stage,
        manifest,
        execution_plan,
        external,
    )
