"""Deterministic construction of heterogeneous per-pipeline Cornstarch meshes."""

from __future__ import annotations

from typing import Any

from oobleck.types import OobleckExecutionPlan


def create_heterogeneous_pipeline_meshes(
    execution_plan: OobleckExecutionPlan,
    *,
    device_type: str,
) -> tuple[dict[str, Any], Any]:
    """Create every pipeline mesh in identical world-wide order.

    ``DeviceMesh`` group creation is world-order-sensitive even for non-members,
    so every rank calls this for every instance. The second return value is the
    current rank's local pipeline mesh.
    """

    import torch.distributed as dist
    from cornstarch.distributed import ModalProcessGroupMesh

    if not dist.is_initialized():
        raise RuntimeError("WORLD must be initialized before mesh activation")
    rank = dist.get_rank()
    meshes: dict[str, Any] = {}
    local = None
    for instance in sorted(execution_plan.instances, key=lambda item: item.instance_id):
        if not instance.ranks:
            raise ValueError(f"pipeline {instance.instance_id} has no concrete rank groups")
        global_ranks = [rank for stage_ranks in instance.ranks for rank in stage_ranks]
        mesh = ModalProcessGroupMesh(
            device_type=device_type,
            global_ranks=global_ranks,
            dp_size=1,
            cp_size=1,
            tp_size=instance.template.tensor_parallel_size,
            num_pp_stages=instance.template.num_stages,
            ep_size=1,
        )
        meshes[instance.instance_id] = mesh
        if rank in global_ranks:
            if local is not None:
                raise ValueError(f"rank {rank} belongs to more than one pipeline")
            local = mesh
    if local is None:
        raise ValueError(f"rank {rank} is not assigned to any pipeline")
    return meshes, local
