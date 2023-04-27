import torch

from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.dataloader import _collate_fn_t
from datasets import Dataset
from typing import Iterator, List
from enum import Enum

from transformers import TrainingArguments


class OobleckSampler(BatchSampler):
    """
    Sampler that generates batches of indices.
    To support heterogeneous pipeline execution, we need to get total number of microbatches
    and number of microbatches for this worker, so that no processor takes the same input.
    """

    def __init__(
        self,
        dataset: Dataset,
        microbatch_size: int,
        num_total_microbatches: int,
        num_my_microbatches: int,
        consumed_samples: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.num_samples = len(dataset)
        self.microbatch_size = microbatch_size
        self.num_microbatches = num_my_microbatches
        self.num_total_microbatches = num_total_microbatches
        self.total_bucket_size = self.microbatch_size * self.num_total_microbatches
        self.consumed_samples = consumed_samples
        self.epoch = epoch

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        self.consumed_samples = 0
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        while self.consumed_samples < self.num_samples:
            if self.num_samples - self.consumed_samples < self.total_bucket_size:
                # Last batch if not complete will be dropped.
                break

            for i in range(self.num_microbatches):
                self.consumed_samples += self.microbatch_size
                yield indices[i * self.microbatch_size : (i + 1) * self.microbatch_size]

    def __len__(self) -> int:
        return self.num_samples // self.total_bucket_size


class LoaderType(Enum):
    Training = (0,)
    Evaluation = (1,)


class OobleckDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        args: TrainingArguments,
        dataloaer_type: LoaderType,
        num_total_microbatches: int,
        consumed_samples: int,
        epoch: int,
        collate_fn: _collate_fn_t,
    ):
        assert isinstance(
            dataset, Dataset
        ), f"dataset type must be datasets.Dataset. Given: {type(dataset)}"

        self.num_total_microbatches = num_total_microbatches
        self.num_my_microbatches = args.gradient_accumulation_steps

        sampler = OobleckSampler(
            dataset,
            args.per_device_train_batch_size
            if dataloaer_type == LoaderType.Training
            else args.per_device_eval_batch_size,
            self.num_total_microbatches,
            self.num_my_microbatches,
            consumed_samples,
            epoch,
        )

        super().__init__(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
