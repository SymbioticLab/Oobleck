"""Version-checked full process-group teardown for generation replacement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable


class ProcessGroupLayoutError(RuntimeError):
    pass


_WORLD_REGISTRIES = (
    "pg_map",
    "pg_names",
    "pg_group_ranks",
    "pg_backend_config",
    "pg_to_tag",
    "tags_to_pg",
    "pg_coalesce_state",
    "pg_default_device",
)


def _shutdown_backend(group: object) -> None:
    shutdown = getattr(group, "_shutdown", None)
    if callable(shutdown):
        shutdown()


def destroy_process_group_universe(extra_handles: Iterable[object] = ()) -> None:
    """Destroy every known group and clear guarded c10d registries.

    This is intentionally strict: an unknown private registry layout is safer
    as a loud compatibility failure than as a silently retained communicator.
    """

    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    handles = list(extra_handles)
    world = getattr(c10d, "_world", None)
    if world is None or any(not hasattr(world, name) for name in _WORLD_REGISTRIES):
        missing = [name for name in _WORLD_REGISTRIES if world is None or not hasattr(world, name)]
        raise ProcessGroupLayoutError(
            f"unsupported PyTorch c10d registry layout; missing {missing}"
        )
    handles.extend(list(world.pg_map))
    unique = list({id(handle): handle for handle in handles}.values())
    if unique:
        with ThreadPoolExecutor(max_workers=len(unique)) as executor:
            list(executor.map(_shutdown_backend, unique))
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    update = getattr(c10d, "_update_default_pg", None)
    if callable(update):
        update(None)
    for name in _WORLD_REGISTRIES:
        getattr(world, name).clear()
    if hasattr(world, "group_count"):
        world.group_count = 0
    if dist.is_initialized():
        raise RuntimeError("WORLD remained initialized after complete teardown")
