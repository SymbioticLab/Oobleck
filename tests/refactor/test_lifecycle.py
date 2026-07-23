from __future__ import annotations

import tempfile

import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from oobleck.distributed import destroy_process_group_universe


def test_world_can_be_fully_destroyed_and_recreated_twice():
    for _ in range(2):
        with tempfile.NamedTemporaryFile() as rendezvous:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous.name}",
                rank=0,
                world_size=1,
            )
            assert dist.is_initialized()
            destroy_process_group_universe()
            assert not dist.is_initialized()
            assert not c10d._world.pg_map
            assert not c10d._world.pg_names
            assert not c10d._world.pg_group_ranks
