"""Version-checked full process-group teardown for generation replacement."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Iterable


class ProcessGroupLayoutError(RuntimeError):
    pass


def initialize_process_group(
    *,
    backend: str,
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    timeout_s: float,
) -> None:
    """Create one validated replacement WORLD from deterministic rendezvous data."""

    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError("WORLD must be fully retired before replacement")
    if not backend or not master_address:
        raise ValueError("backend and rendezvous address are required")
    if not 1 <= master_port <= 65535:
        raise ValueError("rendezvous port must be between 1 and 65535")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid")
    if timeout_s <= 0:
        raise ValueError("rendezvous timeout must be positive")
    host = f"[{master_address}]" if ":" in master_address else master_address
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{host}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        destroy_process_group_universe()
        raise RuntimeError("initialized WORLD does not match the generation rank map")


_REQUIRED = (
    "pg_map",
    "pg_names",
    "pg_group_ranks",
    "pg_backend_config",
    "pg_to_tag",
    "tags_to_pg",
    "pg_coalesce_state",
)
_OPTIONAL = ("_pg_coalesce_state", "pg_default_device")


def _shutdown(group: object) -> None:
    shutdown = getattr(group, "_shutdown", None)
    if callable(shutdown):
        shutdown()


def destroy_process_group_universe(extra_handles: Iterable[object] = ()) -> None:
    """Concurrently stop all groups, destroy WORLD, and clear known registries."""

    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or not (2, 6) <= tuple(map(int, version.groups())) <= (2, 10):
        raise ProcessGroupLayoutError(
            f"PyTorch {torch.__version__} is outside the audited c10d teardown range 2.6-2.10"
        )
    world = getattr(c10d, "_world", None)
    missing = [name for name in _REQUIRED if world is None or not hasattr(world, name)]
    if missing:
        raise ProcessGroupLayoutError(
            f"unsupported PyTorch c10d registry layout; missing {missing}"
        )
    handles = [*extra_handles, *list(world.pg_map)]
    unique = list({id(handle): handle for handle in handles}.values())
    if unique:
        with ThreadPoolExecutor(max_workers=len(unique)) as executor:
            list(executor.map(_shutdown, unique))
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    update = getattr(c10d, "_update_default_pg", None)
    if callable(update):
        update(None)
    for name in (*_REQUIRED, *_OPTIONAL):
        if hasattr(world, name):
            registry = getattr(world, name)
            clear = getattr(registry, "clear", None)
            if callable(clear):
                clear()
    if hasattr(world, "group_count"):
        world.group_count = 0
    if dist.is_initialized():
        raise RuntimeError("WORLD remained initialized after complete teardown")
