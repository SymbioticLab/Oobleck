"""Public replayable DataLoader API with strict attachment validation."""

from __future__ import annotations

from torch.utils.data import DataLoader

from oobleck.data_base import (
    OobleckBatch,
    OobleckBatchDescriptor,
    OobleckBatchSampler,
    PipelineBatchAssignment,
    PreparedDataLoader,
    logical_seed,
)
from oobleck.data_base import prepare_dataloader as _prepare_dataloader


def prepare_dataloader(dataloader: DataLoader) -> PreparedDataLoader:
    sampler = getattr(dataloader, "batch_sampler", None)
    if isinstance(sampler, OobleckBatchSampler) and sampler.dataset is not dataloader.dataset:
        raise ValueError(
            "OobleckBatchSampler was created for a different dataset object than "
            "the DataLoader; attach it to the exact stable map-style dataset"
        )
    return _prepare_dataloader(dataloader)


__all__ = [
    "OobleckBatch",
    "OobleckBatchDescriptor",
    "OobleckBatchSampler",
    "PipelineBatchAssignment",
    "PreparedDataLoader",
    "logical_seed",
    "prepare_dataloader",
]
