"""Replayable logical batches on top of a standard PyTorch DataLoader."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from oobleck.types import PipelineInstance


def logical_seed(seed: int, *keys: object) -> int:
    """Derive a stable 63-bit RNG seed from logical, not physical, identity."""

    digest = hashlib.sha256(
        "\x1f".join([str(seed), *(str(key) for key in keys)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class PipelineBatchAssignment:
    pipeline_id: str
    sample_indices: tuple[int, ...]
    global_microbatch_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OobleckBatchDescriptor:
    epoch: int
    logical_batch_id: int
    sample_indices: tuple[int, ...]
    assignments: tuple[PipelineBatchAssignment, ...]


@dataclass(frozen=True, slots=True)
class OobleckBatch:
    descriptor: OobleckBatchDescriptor
    microbatches: tuple[Any, ...]

    @property
    def sample_indices(self) -> tuple[int, ...]:
        return self.descriptor.sample_indices

    @property
    def logical_batch_id(self) -> int:
        return self.descriptor.logical_batch_id


class OobleckBatchSampler(BatchSampler):
    """Deterministic sampler whose cursor advances only after step commit."""

    def __init__(
        self,
        dataset: object,
        *,
        global_batch_size: int,
        microbatch_size: int,
        instances: Sequence[PipelineInstance],
        seed: int,
        shuffle: bool,
        drop_last: bool = True,
    ) -> None:
        _validate_dataset(dataset)
        if global_batch_size < 1 or microbatch_size < 1:
            raise ValueError("batch sizes must be positive")
        if global_batch_size % microbatch_size:
            raise ValueError("global_batch_size must be divisible by microbatch_size")
        if not instances:
            raise ValueError("at least one pipeline instance is required")
        allocated = sum(item.microbatches for item in instances)
        if allocated != global_batch_size // microbatch_size:
            raise ValueError(
                f"pipeline allocation contains {allocated} microbatches; expected "
                f"{global_batch_size // microbatch_size}"
            )
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.microbatch_size = microbatch_size
        self.instances = tuple(instances)
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self._committed_cursor = 0
        self._issued: dict[int, OobleckBatchDescriptor] = {}

    @property
    def committed_cursor(self) -> int:
        return self._committed_cursor

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self._committed_cursor and epoch != self.epoch:
            raise RuntimeError("cannot change epoch with an unconsumed committed cursor")
        self.epoch = epoch
        self._committed_cursor = 0
        self._issued.clear()

    def _permutation(self) -> list[int]:
        size = len(self.dataset)  # type: ignore[arg-type]
        if not self.shuffle:
            return list(range(size))
        generator = torch.Generator()
        generator.manual_seed(logical_seed(self.seed, "sampler", self.epoch))
        return torch.randperm(size, generator=generator).tolist()

    def descriptor_at(self, batch_index: int) -> OobleckBatchDescriptor:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        logical_id = batch_index
        cached = self._issued.get(logical_id)
        if cached is not None:
            return cached
        permutation = self._permutation()
        start = batch_index * self.global_batch_size
        indices = permutation[start : start + self.global_batch_size]
        if len(indices) < self.global_batch_size and self.drop_last:
            raise IndexError(batch_index)
        assignments = []
        sample_cursor = 0
        microbatch_cursor = 0
        for instance in self.instances:
            sample_count = instance.microbatches * self.microbatch_size
            assigned = tuple(indices[sample_cursor : sample_cursor + sample_count])
            assignments.append(
                PipelineBatchAssignment(
                    instance.instance_id,
                    assigned,
                    tuple(
                        range(
                            microbatch_cursor,
                            microbatch_cursor + instance.microbatches,
                        )
                    ),
                )
            )
            sample_cursor += sample_count
            microbatch_cursor += instance.microbatches
        descriptor = OobleckBatchDescriptor(
            self.epoch, logical_id, tuple(indices), tuple(assignments)
        )
        self._issued[logical_id] = descriptor
        return descriptor

    def __iter__(self) -> Iterator[list[int]]:
        # Prefetch may advance this iterator arbitrarily far.  The committed
        # cursor is deliberately untouched and a fresh iterator starts there.
        for index in range(self._committed_cursor, len(self)):
            yield list(self.descriptor_at(index).sample_indices)

    def __len__(self) -> int:
        size = len(self.dataset)  # type: ignore[arg-type]
        if self.drop_last:
            return size // self.global_batch_size
        return (size + self.global_batch_size - 1) // self.global_batch_size

    def commit(self, descriptor: OobleckBatchDescriptor) -> None:
        if descriptor.epoch != self.epoch:
            raise RuntimeError("cannot commit a batch from a stale epoch")
        if descriptor.logical_batch_id != self._committed_cursor:
            raise RuntimeError(
                f"out-of-order batch commit: got {descriptor.logical_batch_id}, "
                f"expected {self._committed_cursor}"
            )
        expected = self.descriptor_at(self._committed_cursor)
        if descriptor != expected:
            raise RuntimeError("batch descriptor does not match deterministic sampler state")
        self._committed_cursor += 1
        self._issued.pop(descriptor.logical_batch_id, None)

    def rewind_uncommitted(self) -> None:
        self._issued = {
            key: value for key, value in self._issued.items() if key < self._committed_cursor
        }

    def reconfigure(self, instances: Sequence[PipelineInstance]) -> None:
        selected = tuple(instances)
        allocated = sum(item.microbatches for item in selected)
        expected = self.global_batch_size // self.microbatch_size
        if not selected or allocated != expected:
            raise ValueError(
                f"new pipeline allocation contains {allocated} microbatches; expected {expected}"
            )
        self.instances = selected
        self.rewind_uncommitted()


def _validate_dataset(dataset: object) -> None:
    if isinstance(dataset, IterableDataset):
        raise TypeError(
            "Oobleck requires a stable map-style Dataset; streaming and other "
            "IterableDataset instances need a durable cursor protocol"
        )
    if dataset.__class__.__name__ == "DatasetDict" or (
        isinstance(dataset, Mapping)
        and all(isinstance(key, str) for key in dataset)
        and not hasattr(dataset, "column_names")
    ):
        raise TypeError(
            "A DatasetDict is not a sample dataset. Select a split, for example "
            "dataset['train'] or load_dataset(..., split='train')."
        )
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("dataset must implement stable __len__ and indexed __getitem__")


def _slice_collated(value: Any, start: int, end: int, total: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_collated(item, start, end, total) for key, item in value.items()}
    if isinstance(value, tuple) and len(value) == total:
        return value[start:end]
    if isinstance(value, list) and len(value) == total:
        return value[start:end]
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == total:
        return value[start:end]
    return value


def _normalize_microbatches(
    collated: Any, descriptor: OobleckBatchDescriptor, microbatch_size: int
) -> tuple[Any, ...]:
    count = sum(len(item.global_microbatch_ids) for item in descriptor.assignments)
    if isinstance(collated, list) and len(collated) == count:
        return tuple(collated)
    total = len(descriptor.sample_indices)
    return tuple(
        _slice_collated(
            collated,
            index * microbatch_size,
            min((index + 1) * microbatch_size, total),
            total,
        )
        for index in range(count)
    )


class PreparedDataLoader:
    """Ordered adapter that pairs DataLoader results with logical descriptors."""

    def __init__(self, dataloader: DataLoader, sampler: OobleckBatchSampler) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self._invalidated = False

    def invalidate_prefetch(self) -> None:
        self._invalidated = True
        self.sampler.rewind_uncommitted()

    def reconfigure(self, instances: Sequence[PipelineInstance]) -> None:
        self.invalidate_prefetch()
        self.sampler.reconfigure(instances)

    def __iter__(self) -> Iterator[OobleckBatch]:
        self._invalidated = False
        start = self.sampler.committed_cursor
        for offset, collated in enumerate(iter(self.dataloader)):
            if self._invalidated:
                return
            descriptor = self.sampler.descriptor_at(start + offset)
            yield OobleckBatch(
                descriptor,
                _normalize_microbatches(collated, descriptor, self.sampler.microbatch_size),
            )

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.sampler.committed_cursor)


def prepare_dataloader(dataloader: DataLoader) -> PreparedDataLoader:
    if not isinstance(dataloader, DataLoader):
        raise TypeError("prepare_dataloader expects torch.utils.data.DataLoader")
    _validate_dataset(dataloader.dataset)
    sampler = dataloader.batch_sampler
    if not isinstance(sampler, OobleckBatchSampler):
        raise TypeError(
            "DataLoader must use the OobleckBatchSampler returned by "
            "context.create_batch_sampler() via batch_sampler="
        )
    if dataloader.persistent_workers:
        raise ValueError(
            "persistent_workers must be False so failed prefetch workers can be "
            "discarded and deterministically recreated"
        )
    generator = torch.Generator()
    generator.manual_seed(logical_seed(sampler.seed, "dataloader", sampler.epoch))
    dataloader.generator = generator
    return PreparedDataLoader(dataloader, sampler)
