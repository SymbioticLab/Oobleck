from enum import Enum
from typing import Iterator, List

import torch
from datasets import Dataset
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.dataloader import _collate_fn_t
from transformers.training_args import TrainingArguments

from oobleck.execution.dataset import OobleckDataset


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

        """
        Microbatches for one iteration are contiguous for each data parallel pipeline.
        Once an iteration is done, it jumps to the next contiguous microbatch
        that is not consumed by other data parallel pipelines.
        """
        while self.consumed_samples < self.num_samples:
            if self.num_samples - self.consumed_samples < self.total_bucket_size:
                # Last batch if not complete will be dropped.
                break

            for i in range(self.num_microbatches):
                self.consumed_samples += self.microbatch_size
                yield indices[i * self.microbatch_size : (i + 1) * self.microbatch_size]

            self.consumed_samples += self.microbatch_size * (
                self.num_total_microbatches - self.num_microbatches
            )

    def __len__(self) -> int:
        return self.num_samples // self.total_bucket_size


class LoaderType(Enum):
    Training = (0,)
    Evaluation = (1,)


class OobleckDataLoader(DataLoader):
    def __init__(
        self,
        datasets: OobleckDataset,
        args: TrainingArguments,
        dataloader_type: LoaderType,
        num_total_microbatches: int,
        consumed_samples: int,
        epoch: int,
        shuffle: bool = True,
    ):
        assert isinstance(
            datasets, OobleckDataset
        ), f"dataset type must be OobleckDataset. Given: {type(dataset)}"

        self.num_total_microbatches = num_total_microbatches
        self.num_my_microbatches = args.gradient_accumulation_steps

        if dataloader_type == LoaderType.Training:
            dataset = datasets.dataset["train"]
            batch_size = args.per_device_train_batch_size
        else:
            dataset = datasets.dataset["validation"]
            batch_size = args.per_device_eval_batch_size

        sampler = OobleckSampler(
            dataset,
            batch_size,
            self.num_total_microbatches,
            self.num_my_microbatches,
            consumed_samples,
            epoch,
            shuffle,
        )

        super().__init__(
            dataset,
            batch_sampler=sampler,
            collate_fn=datasets.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
