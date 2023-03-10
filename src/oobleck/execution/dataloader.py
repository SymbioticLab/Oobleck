import torch

from torch.utils.data import Sampler, DataLoader
from torch.utils.data.dataset import T_co
from torch.utils.data.dataloader import _collate_fn_t
from datasets import Dataset
from typing import Iterator

from transformers import TrainingArguments


class OobleckSampler(Sampler[T_co]):
    def __init__(
        self,
        num_samples: int,
        microbatch_size: int,
        consumed_samples: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.microbatch_size = microbatch_size
        self.num_microbatches = 4  # TODO: modify it
        self.total_bucket_size = self.microbatch_size * self.num_microbatches
        self.consumed_samples = consumed_samples
        self.epoch = epoch

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
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
                return

            for i in range(self.num_microbatches):
                yield indices[i * self.microbatch_size : (i + 1) * self.microbatch_size]

            self.consumed_samples += self.total_bucket_size


class OobleckDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, collate_fn: _collate_fn_t, args: TrainingArguments
    ):
        sampler = OobleckSampler(dataset, args.per_device_train_batch_size)

        super().__init__(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
