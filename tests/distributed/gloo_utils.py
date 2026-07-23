# Compatibility wrapper for the attributed Cornstarch Gloo helpers.
# See tests/distributed/CORNSTARCH_NOTICE.md.

"""PyTorch-version compatibility wrappers over the attributed Gloo helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from tests.distributed._gloo_utils_base import (
    all_gather_gloo,
    all_to_all_gloo,
    all_to_all_single_gloo as _all_to_all_single_gloo,
    batch_isend_irecv_gloo,
    reduce_scatter_gloo,
)


def all_to_all_single_gloo(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Optional[list[int]] = None,
    input_split_sizes: Optional[list[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    # PyTorch 2.10 no longer accepts None in get_global_rank().  The helper's
    # subgroup translation is still correct; make WORLD explicit for that call.
    selected_group = dist.GroupMember.WORLD if group is None else group
    return _all_to_all_single_gloo(
        output,
        input,
        output_split_sizes,
        input_split_sizes,
        selected_group,
        async_op,
    )


__all__ = [
    "all_gather_gloo",
    "all_to_all_gloo",
    "all_to_all_single_gloo",
    "batch_isend_irecv_gloo",
    "reduce_scatter_gloo",
]
