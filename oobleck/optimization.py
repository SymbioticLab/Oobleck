"""Versioned standard-PyTorch optimizer state keyed by logical parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from oobleck.state import LogicalStateEntry


class OptimizerSchemaError(TypeError):
    """An optimizer cannot be represented by Oobleck's supported schema."""


@dataclass(frozen=True, slots=True)
class OptimizerStateSchema:
    """Portable optimizer metadata and versioned logical tensor entries."""

    optimizer_class: str
    parameter_groups: tuple[dict[str, Any], ...]
    scalar_slots: tuple[tuple[str, str, Any], ...]
    tensor_entries: tuple[LogicalStateEntry, ...]
    schema_version: int = 1


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Extract detached rank-local storage from a Tensor or DTensor slot."""

    local = value.to_local() if hasattr(value, "to_local") else value
    return local.detach()


def serialize_optimizer_state(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, torch.nn.Parameter],
    *,
    owner_rank: int,
    committed_step: int,
    parameter_entries: Mapping[str, LogicalStateEntry] | None = None,
) -> tuple[OptimizerStateSchema, dict[str, torch.Tensor]]:
    """Serialize standard optimizer state without unstable parameter IDs.

    ``parameter_entries`` supplies the active TP shard identity.  It is optional
    for the homogeneous compatibility path, but elastic callers always pass it.
    """

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
        parameter_entry = (parameter_entries or {}).get(parameter_key)
        for slot, value in sorted(state.items()):
            logical_key = f"{parameter_key}::optimizer::{slot}"
            if isinstance(value, torch.Tensor):
                tensor = _local_tensor(value)
                tensors[logical_key] = tensor
                same_shape = parameter_entry is not None and tuple(tensor.shape) == tuple(
                    parameter_entry.local_shape
                )
                entries.append(
                    LogicalStateEntry(
                        logical_key,
                        parameter_entry.global_shape if same_shape else tuple(tensor.shape),
                        tuple(tensor.shape),
                        str(tensor.dtype),
                        "optimizer",
                        parameter_entry.placements if same_shape else ("replicate",),
                        parameter_entry.tp_lane if parameter_entry is not None else 0,
                        owner_rank,
                        committed_step,
                        parameter_entry.global_layer_id if parameter_entry is not None else None,
                        parameter_entry.shared_state_id if parameter_entry is not None else None,
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


def optimizer_schema_for_partition(
    schemas: Sequence[OptimizerStateSchema],
    parameter_entries: Mapping[str, LogicalStateEntry],
    *,
    owner_rank: int,
    committed_step: int,
) -> OptimizerStateSchema:
    """Derive optimizer metadata and slots owned by one replacement partition.

    Surviving schemas must agree on optimizer class, schema version, parameter-group count,
    group metadata, and each parameter's group assignment. The new model manifest selects
    only locally owned parameters and matching TP-lane tensor slots; scalar slots are copied
    by logical parameter key. Owners and committed versions are rewritten for the new rank,
    producing a schema that can be planned alongside model state without unstable object IDs.
    """

    if not schemas:
        raise OptimizerSchemaError("no surviving optimizer schema is available")
    optimizer_classes = {schema.optimizer_class for schema in schemas}
    versions = {schema.schema_version for schema in schemas}
    group_counts = {len(schema.parameter_groups) for schema in schemas}
    if len(optimizer_classes) != 1 or versions != {1} or len(group_counts) != 1:
        raise OptimizerSchemaError("surviving optimizer schemas are incompatible")

    group_count = next(iter(group_counts))
    group_metadata: list[dict[str, Any] | None] = [None] * group_count
    parameter_group: dict[str, int] = {}
    for schema in schemas:
        for index, group in enumerate(schema.parameter_groups):
            metadata = {key: value for key, value in group.items() if key != "params"}
            if group_metadata[index] is None:
                group_metadata[index] = metadata
            elif group_metadata[index] != metadata:
                raise OptimizerSchemaError("optimizer parameter-group metadata diverged")
            for key in group["params"]:
                previous = parameter_group.setdefault(key, index)
                if previous != index:
                    raise OptimizerSchemaError(f"parameter {key!r} changed optimizer groups")

    groups = []
    for index, metadata in enumerate(group_metadata):
        group = dict(metadata or {})
        group["params"] = [key for key in parameter_entries if parameter_group.get(key) == index]
        groups.append(group)
    missing = [key for key in parameter_entries if key not in parameter_group]
    if missing:
        raise OptimizerSchemaError(
            f"new partition parameters are absent from optimizer metadata: {missing[:3]}"
        )

    scalars: dict[tuple[str, str], Any] = {}
    entries: dict[tuple[str, int], LogicalStateEntry] = {}
    marker = "::optimizer::"
    for schema in schemas:
        for parameter_key, slot, value in schema.scalar_slots:
            if parameter_key in parameter_entries:
                scalars.setdefault((parameter_key, slot), value)
        for entry in schema.tensor_entries:
            parameter_key = entry.logical_key.split(marker, 1)[0]
            parameter_entry = parameter_entries.get(parameter_key)
            if parameter_entry is None or entry.tp_lane != parameter_entry.tp_lane:
                continue
            identity = (entry.logical_key, entry.tp_lane)
            entries.setdefault(
                identity,
                replace(entry, owner_rank=owner_rank, version=committed_step),
            )

    return OptimizerStateSchema(
        next(iter(optimizer_classes)),
        tuple(groups),
        tuple((key, slot, value) for (key, slot), value in sorted(scalars.items())),
        tuple(entries[key] for key in sorted(entries)),
    )


def _restore_tensor_for_parameter(
    tensor: torch.Tensor,
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    """Restore local storage and rewrap it as DTensor when the parameter is sharded."""

    local = tensor.to(parameter.device).clone()
    if not hasattr(parameter, "device_mesh") or tuple(local.shape) != tuple(
        getattr(parameter, "_local_tensor", local).shape
    ):
        return local
    try:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            local,
            parameter.device_mesh,
            parameter.placements,
            run_check=False,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return local


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    schema: OptimizerStateSchema,
    tensors: Mapping[str, torch.Tensor],
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> None:
    """Rebuild optimizer groups and state using stable logical parameter identities.

    Class/schema and group-count checks prevent recovery into a semantically different
    optimizer. Parameter lists are rebound to the replacement model, group options and scalar
    slots are restored, and every declared tensor slot must be present. Sharded parameters
    rewrap matching local storage as DTensor when possible; ordinary slots remain local tensors
    on the parameter device.
    """

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
        optimizer.state[parameter][slot] = _restore_tensor_for_parameter(
            tensors[entry.logical_key], parameter
        )


__all__ = [
    "OptimizerSchemaError",
    "OptimizerStateSchema",
    "optimizer_schema_for_partition",
    "restore_optimizer_state",
    "serialize_optimizer_state",
]
