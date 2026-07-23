from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from oobleck import (
    build_runtime_compatibility,
    dataset_fingerprint,
    model_fingerprint,
)


class StableDataset(Dataset):
    oobleck_fingerprint = "stable-v1"

    def __init__(self, length: int = 4) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)


class UnstableDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return index


class Stream(IterableDataset):
    def __iter__(self):
        yield 1


def test_runtime_compatibility_is_stable_and_sensitive_to_inputs():
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 2)
    first = build_runtime_compatibility(
        model,
        StableDataset(),
        model_identity="linear-checkpoint-v1",
        oobleck_revision="oobleck-sha",
        cornstarch_revision="cornstarch-sha",
    )
    second = build_runtime_compatibility(
        model,
        StableDataset(),
        model_identity="linear-checkpoint-v1",
        oobleck_revision="oobleck-sha",
        cornstarch_revision="cornstarch-sha",
    )
    assert first == second
    assert first.digest == second.digest
    assert first.model_fingerprint == model_fingerprint(
        model, model_identity="linear-checkpoint-v1"
    )
    changed = dataset_fingerprint(StableDataset(), preprocessing="tokenizer-v2")
    assert changed != first.dataset_fingerprint
    assert dataset_fingerprint(StableDataset(5)) != first.dataset_fingerprint


def test_dataset_compatibility_requires_stable_map_identity():
    with pytest.raises(ValueError, match="no stable fingerprint"):
        dataset_fingerprint(UnstableDataset())
    with pytest.raises(TypeError, match="map-style"):
        dataset_fingerprint(Stream())
    assert dataset_fingerprint(UnstableDataset(), explicit="provided")
