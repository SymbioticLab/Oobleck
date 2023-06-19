import os
from typing import List

import pytest
import torch

from oobleck.execution.dataloader import OobleckDataLoader, OobleckSampler
from tests.conftest import (
    GRADIENT_ACCUMULATION_STEP,
    TRAIN_BATCH_SIZE,
    OobleckDynamicClassFactory,
    OobleckMultiProcessTestCase,
    OobleckStaticClassFactory,
)


class TestOobleckDataloader(OobleckMultiProcessTestCase):
    @staticmethod
    def _attributes(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_microbatches: List[int],
        num_iterations: int,
    ):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dataloader: OobleckDataLoader = dfactory.get_dataloader(
            rank, num_microbatches, num_iterations
        )
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.microbatch_size == TRAIN_BATCH_SIZE
        assert sampler.num_iterations_done == num_iterations
        assert len(sampler.num_microbatches) == world_size
        assert sampler.num_microbatches[rank] == GRADIENT_ACCUMULATION_STEP

    @pytest.mark.parametrize("num_iterations", [0, 14])
    def test_attributes_type(self, num_iterations: int):
        self.run_in_parallel(
            4,
            TestOobleckDataloader._attributes,
            [GRADIENT_ACCUMULATION_STEP] * 4,
            num_iterations,
        )

    @staticmethod
    def _batch(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_microbatches: List[int],
    ):
        rank = int(os.environ["RANK"])
        dataloader: OobleckDataLoader = dfactory.get_dataloader(rank, num_microbatches)
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.num_iterations_done == 0

        inputs = next(iter(dataloader))
        assert isinstance(inputs, dict)
        for tensor in inputs.values():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.size(dim=0) == TRAIN_BATCH_SIZE

    def test_batch_samples(self):
        self.run_in_parallel(
            1,
            TestOobleckDataloader._batch,
            [1],
        )

    @staticmethod
    def _iteration(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_microbatches: List[int],
    ):
        rank = int(os.environ["RANK"])
        dataloader: OobleckDataLoader = dfactory.get_dataloader(rank, num_microbatches)
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.num_iterations_done == 0

        iterator = iter(dataloader)

        # Get a set of batches for one iteration
        # Within the iteration they should be contiguous
        for i in range(GRADIENT_ACCUMULATION_STEP):
            next(iterator)

        assert sampler.num_iterations_done == 1

    def test_distributed_batch_samples(self):
        self.run_in_parallel(
            4,
            TestOobleckDataloader._iteration,
            [GRADIENT_ACCUMULATION_STEP] * 4,
        )

    @pytest.mark.skip(reason="Not implemented yet")
    def test_unique_batches_per_index(self):
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_batch_deterministric(self):
        pass

    @staticmethod
    def _run_until_stop(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        num_microbatches: List[int],
    ):
        rank = int(os.environ["RANK"])

        dataloader: OobleckDataLoader = dfactory.get_dataloader(rank, num_microbatches)
        sampler: OobleckSampler = dataloader.batch_sampler
        iterator = iter(dataloader)
        assert sampler.num_iterations_done == 0
        assert sampler.epoch == 0

        try:
            for iter_num in range(len(dataloader)):
                assert sampler.num_iterations_done == iter_num
                for _ in range(num_microbatches[rank]):
                    next(iterator)
                assert sampler.num_iterations_done == iter_num + 1
        except StopIteration:
            raise AssertionError("StopIteration raised before the last iteration.")

        with pytest.raises(StopIteration):
            next(iterator)

        assert sampler.num_iterations_done == 0
        assert sampler.epoch == 1

    def test_stop_iteration(self):
        self.run_in_parallel(
            1,
            TestOobleckDataloader._run_until_stop,
            [GRADIENT_ACCUMULATION_STEP],
        )
