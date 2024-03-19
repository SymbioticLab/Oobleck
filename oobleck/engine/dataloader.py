from typing import Iterator, cast

import torch
from oobleck_colossalai.plugin.heterogeneous_dataloader import (
    HeterogeneousBatchSampler,
    HeterogeneousDataLoader,
)
from torch.utils.data import DataLoader, Dataset


class OobleckBatchSampler(HeterogeneousBatchSampler):
    """
    Same with HeterogenouesBatchSampler, but cache the previous element
    so that when failure happens it can go back to the previous element.
    """

    def __init__(
        self,
        dataset: Dataset,
        pipeline_index: int,
        microbatch_size: int,
        num_microbatches: list[int],
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sample_index: int = 0,
    ):
        super().__init__(
            dataset,
            pipeline_index,
            microbatch_size,
            num_microbatches,
            shuffle,
            seed,
            drop_last,
        )

        self.sample_index = sample_index

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        """
        Because global batch size is not changed across sampler reinstantiation,
        the number of iteration (len(self.num_samples)) should also not be changed,
        but just the pipeline index and num_microbatches per pipeline.
        """
        while self.sample_index < len(self.num_samples):
            index = indices[self.sample_index] * self.global_batch_size
            # increment the sample index for the next iteration.
            # If training fails in the current iteration, will be decreased by 1 by step_back()
            # to retrain the same iteration.
            self.sample_index += 1

            # batch start indices for the current pipeline within this iteration
            pipeline_batch_offset = sum(self.num_microbatches[: self.pipeline_index])

            # Return all microbatches for the current iteration at once.
            # This is because the current ColossalAI implementation fetches all microbatches.
            batch_start_index = index + self.microbatch_size * pipeline_batch_offset
            yield list(
                range(
                    batch_start_index,
                    batch_start_index
                    + self.microbatch_size * self.num_microbatches[self.pipeline_index],
                )
            )

        # Reset index after iterating all samples
        # After returning, the iterator will return StopIteration;
        # and when a new iterator is created, it will start from the beginning.
        # If an iterator is created while the sample_index is not reset,
        # this means a failure happens in the middle and should be resumed from there.
        self.sample_index = 0

    def __len__(self) -> int:
        return self.num_samples


class OobleckDataLoader(HeterogeneousDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        global_batch_size: int = 1,
        microbatch_size: int = 1,
        shuffle: bool = True,
        seed: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        kwargs["drop_last"] = True
        super().__init__(
            dataset=dataset,
            global_batch_size=global_batch_size,
            microbatch_size=microbatch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    def configure(self, pipeline_index: int, num_microbatches: list[int]):
        assert self.global_batch_size == self.microbatch_size * sum(num_microbatches), (
            f"inconsistent global batch size={self.global_batch_size}, microbatch_size={self.microbatch_size}, "
            f"and num_microbatches={num_microbatches}"
        )

        batch_sampler = OobleckBatchSampler(
            self.dataset,
            pipeline_index,
            self.microbatch_size,
            num_microbatches,
            self.shuffle,
            self.seed,
            self.drop_last,
        )
        DataLoader.__init__(
            self,
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self.kwargs,
        )

    def reconfigure(self, pipeline_index: int, num_microbatches: list[int]):
        batch_sampler: OobleckBatchSampler = cast(
            OobleckBatchSampler, self.batch_sampler
        )
        sample_index = (
            batch_sampler.sample_index - 1 if batch_sampler.sample_index > 0 else 0
        )

        self.__initialized = False

        batch_sampler = OobleckBatchSampler(
            self.dataset,
            pipeline_index,
            self.microbatch_size,
            num_microbatches,
            self.shuffle,
            self.seed,
            self.drop_last,
            sample_index=sample_index,
        )
        DataLoader.__init__(
            self,
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self.kwargs,
        )
