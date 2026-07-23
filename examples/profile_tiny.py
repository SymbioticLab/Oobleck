"""Offline profiling workload used by the operator walkthrough."""

from __future__ import annotations

import torch

from examples.pretrain_llm_base import TinyLanguageModel
from oobleck.config import ProfileCommandConfig
from oobleck.planning import ProfilingWorkload


def create_workload(config: ProfileCommandConfig) -> ProfilingWorkload:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config.dtype)
    model = TinyLanguageModel().to(device=device, dtype=dtype)
    model.embedding._oobleck_profile_name = "embedding"
    model.output._oobleck_profile_name = "output"

    def inputs(layer_index: int, iteration: int):
        if layer_index == 0:
            return torch.zeros(
                config.microbatch_size,
                15,
                dtype=torch.long,
                device=device,
            )
        return torch.ones(
            config.microbatch_size,
            15,
            model.output.in_features,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    return ProfilingWorkload((model.embedding, model.output), inputs)


__all__ = ["create_workload"]
