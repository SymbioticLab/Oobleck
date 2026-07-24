"""Versioned standard-PyTorch optimizer state schema keyed by logical parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from oobleck.state import LogicalStateEntry


class OptimizerSchemaError(TypeError):
    """An optimizer cannot be represented by the retained logical-key schema."""


@dataclass(frozen=True, slots=True)
class OptimizerStateSchema:
    """Optimizer class, parameter groups, scalar slots, and tensor manifests."""

    optimizer_class: str
    parameter_groups: tuple[dict[str, Any], ...]
    scalar_slots: tuple[tuple[str, str, Any], ...]
    tensor_entries: tuple[LogicalStateEntry, ...]
    schema_version: int = 1


def serialize_optimizer_state(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, torch.nn.Parameter],
    *,
    owner_rank: int,
    committed_step: int,
) -> tuple[OptimizerStateSchema, dict[str, torch.Tensor]]:
    """Replace unstable parameter IDs with logical keys and versioned slot tensors."""

    by_identity = {id(parameter): name for name, parameter in named_parameters.items()}
    groups = []
    for group in optimizer.param_groups:
        try:
            keys = [by_identity[id(parameter)] for parameter in group["params"]]
        except KeyError as exc:
            raise OptimizerSchemaError(
                "optimizer owns a parameter without a logical model key"
            ) from exc
        metadata = {key: value for key, value in group.items() if key != "params"}
        metadata["params"] = keys
        groups.append(metadata)

    entries = []
    tensors: dict[str, torch.Tensor] = {}
    scalars = []
    primitive = (str, int, float, bool, type(None))
    for parameter, state in optimizer.state.items():
        parameter_key = by_identity.get(id(parameter))
        if parameter_key is None:
            raise OptimizerSchemaError("optimizer state has no stable logical parameter key")
        for slot, value in sorted(state.items()):
            logical_key = f"{parameter_key}::optimizer::{slot}"
            if isinstance(value, torch.Tensor):
                tensor = value.detach()
                tensors[logical_key] = tensor
                entries.append(
                    LogicalStateEntry(
                        logical_key,
                        tuple(tensor.shape),
                        tuple(tensor.shape),
                        str(tensor.dtype),
                        "optimizer",
                        ("replicate",),
                        0,
                        owner_rank,
                        committed_step,
                    )
                )
            elif isinstance(value, primitive):
                scalars.append((parameter_key, str(slot), value))
            else:
                raise OptimizerSchemaError(
                    f"optimizer slot {parameter_key}:{slot} has unsupported "
                    f"value type {type(value).__name__}"
                )
    schema = OptimizerStateSchema(
        f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
        tuple(groups),
        tuple(scalars),
        tuple(entries),
    )
    return schema, tensors


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    schema: OptimizerStateSchema,
    tensors: Mapping[str, torch.Tensor],
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> None:
    """Reconstruct parameter groups and slots into a compatible optimizer."""

    actual = f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
    if schema.schema_version != 1 or actual != schema.optimizer_class:
        raise OptimizerSchemaError(
            f"optimizer schema mismatch: serialized={schema.optimizer_class}, actual={actual}"
        )
    if len(schema.parameter_groups) != len(optimizer.param_groups):
        raise OptimizerSchemaError("optimizer parameter-group count changed during recovery")
    optimizer.state.clear()
    for destination, saved in zip(optimizer.param_groups, schema.parameter_groups):
        keys = saved["params"]
        destination["params"] = [named_parameters[key] for key in keys]
        for key, value in saved.items():
            if key != "params":
                destination[key] = value
    for parameter_key, slot, value in schema.scalar_slots:
        optimizer.state[named_parameters[parameter_key]][slot] = value
    for entry in schema.tensor_entries:
        marker = "::optimizer::"
        parameter_key, slot = entry.logical_key.split(marker, 1)
        if entry.logical_key not in tensors:
            raise OptimizerSchemaError(f"missing optimizer tensor {entry.logical_key}")
        parameter = named_parameters[parameter_key]
        optimizer.state[parameter][slot] = tensors[entry.logical_key].to(parameter.device).clone()
