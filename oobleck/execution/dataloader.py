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
        pipeline_index: int,
        num_microbatches: List[int],
        num_iterations_done,
        epoch: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.num_samples = len(dataset)
        self.microbatch_size = microbatch_size
        self.pipeline_index = pipeline_index
        self.num_microbatches = num_microbatches
        self.num_iterations_done = num_iterations_done
        self.epoch = epoch
        self.shuffle = shuffle
        self.seed = seed

        assert self.pipeline_index < len(self.num_microbatches)
        self.total_bucket_size = self.microbatch_size * sum(self.num_microbatches)

    def __iter__(self) -> Iterator[List[int]]:
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
        num_iterations_per_epoch = len(self)

        """
        N == len(num_microbatches)
        n == num_microbatches[rank]
                <-----> microbatch_size
        <------------microbatch_offset---------------> (e.g. if my rank is 2)
        <-------------------------num_jump_microbatches-------------------------->
        [r0mb0|r0mb1|...|r0mbn|r1mb0|r1mb1|...|r1mbn'|...|rNmb0|rNmb1|...|rNmbn'']
        """
        # For each iteration, it should jump this amount of microbatches
        num_jump_microbatches = self.microbatch_size * sum(self.num_microbatches)
        microbatches_offset = (
            sum(self.num_microbatches[: self.pipeline_index]) * self.microbatch_size
        )

        for iter_index in range(num_iterations_per_epoch):
            if (
                self.num_samples - iter_index * num_jump_microbatches
                < self.total_bucket_size
            ):
                # Last batch if not complete will be dropped.
                break

            for mb in range(self.num_microbatches[self.pipeline_index]):
                # increase num iteraion done if it is the last microbatch
                if mb == self.num_microbatches[self.pipeline_index] - 1:
                    self.num_iterations_done += 1

                # TODO:adjust indices corresponding to my pipeline index
                yield indices[
                    iter_index * num_jump_microbatches
                    + mb * self.microbatch_size
                    + microbatches_offset : iter_index * num_jump_microbatches
                    + (mb + 1) * self.microbatch_size
                    + microbatches_offset
                ]

        # After one epoch is done, adjust epoch and num iterations done
        self.num_iterations_done = 0
        self.epoch += 1

    def __len__(self) -> int:
        return self.num_samples // self.total_bucket_size


class LoaderType(Enum):
    Training = (0,)
    Evaluation = (1,)


class OobleckDataLoader(DataLoader):
    def __init__(
        self,
        args: TrainingArguments,
        datasets: OobleckDataset,
        dataloader_type: LoaderType,
        pipeline_index: int,
        num_microbatches: List[int],
        num_iterations_done: int,
        epoch: int,
        shuffle: bool = True,
    ):
        assert isinstance(
            datasets, OobleckDataset
        ), f"dataset type must be OobleckDataset. Given: {type(dataset)}"

        if dataloader_type == LoaderType.Training:
            dataset = datasets.dataset["train"]
            microbatch_size = args.per_device_train_batch_size
        else:
            dataset = datasets.dataset["validation"]
            microbatch_size = args.per_device_eval_batch_size

        sampler = OobleckSampler(
            dataset,
            microbatch_size,
            pipeline_index,
            num_microbatches,
            num_iterations_done,
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
