import pytest
import torch

from oobleck.state import LogicalStateEntry, StateManifest
from oobleck.topology import (
    build_gradient_sync_groups,
    parameter_bindings_by_sync_key,
)


def test_gradient_groups_match_logical_tp_shards_and_sample_weights():
    def manifest(rank, lane):
        return StateManifest(
            rank,
            (
                LogicalStateEntry(
                    "layer.weight",
                    (8, 8),
                    (4, 8),
                    "float32",
                    "parameter",
                    ("shard:0",),
                    lane,
                    rank,
                    0,
                ),
            ),
            0,
        )

    groups = build_gradient_sync_groups(
        [manifest(0, 0), manifest(1, 0), manifest(2, 1)],
        {0: "fast", 1: "slow", 2: "fast"},
        {"fast": 6, "slow": 2},
    )
    assert len(groups) == 2
    assert groups[0].ranks == (0, 1)
    assert groups[0].sample_weights == (0.75, 0.25)
    assert groups[1].ranks == (2,)
    assert groups[1].sample_weights == (1.0,)


def test_tied_parameters_bind_through_shared_state_identity():
    parameter = torch.nn.Parameter(torch.ones(2))

    def entry(key):
        return LogicalStateEntry(
            key,
            (2,),
            (2,),
            "float32",
            "parameter",
            ("replicate",),
            0,
            0,
            0,
            shared_state_id="shared.embedding",
        )

    manifest = StateManifest(0, (entry("embed.weight"), entry("lm_head.weight")), 0)
    bindings = parameter_bindings_by_sync_key(
        {"embed.weight": parameter, "lm_head.weight": parameter}, manifest
    )
    assert bindings["shared.embedding"] is parameter

    with pytest.raises(RuntimeError, match="distinct local tensors"):
        parameter_bindings_by_sync_key(
            {
                "embed.weight": parameter,
                "lm_head.weight": torch.nn.Parameter(torch.zeros(2)),
            },
            manifest,
        )
