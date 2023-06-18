import pytest
import torch

from oobleck.execution.dataloader import LoaderType, OobleckDataLoader, OobleckSampler
from tests.conftest import (
    EVAL_BATCH_SIZE,
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
        total_num_microbatches: int,
        consumed_samples: int,
    ):
        dataloader = dfactory.get_dataloader(total_num_microbatches, consumed_samples)
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.microbatch_size == TRAIN_BATCH_SIZE
        assert sampler.consumed_samples == consumed_samples
        assert sampler.num_microbatches == GRADIENT_ACCUMULATION_STEP
        assert sampler.num_total_microbatches == total_num_microbatches

    @pytest.mark.parametrize("consumed_samples", [0, 40])
    def test_attributes_type(self, consumed_samples):
        self.run_in_parallel(
            4,
            TestOobleckDataloader._attributes,
            4 * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEP,
            consumed_samples,
        )

    @staticmethod
    def _batch(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        total_num_microbatches: int,
    ):
        dataloader = dfactory.get_dataloader(total_num_microbatches)
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.consumed_samples == 0

        inputs = next(iter(dataloader))
        assert sampler.consumed_samples == TRAIN_BATCH_SIZE
        assert isinstance(inputs, dict)
        for tensor in inputs.values():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.size(dim=0) == TRAIN_BATCH_SIZE

    def test_batch_samples(self):
        self.run_in_parallel(
            1,
            TestOobleckDataloader._batch,
            TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEP,
        )

    def test_distributed_batch_samples(self):
        pass

    @staticmethod
    def run_until_stop(
        factory: OobleckStaticClassFactory,
        dfactory: OobleckDynamicClassFactory,
        total_num_microbatches: int,
    ):
        dataloader = dfactory.get_dataloader(total_num_microbatches)
        sampler: OobleckSampler = dataloader.batch_sampler
        assert sampler.consumed_samples == 0

        num_iteration = len(dataloader) * GRADIENT_ACCUMULATION_STEP
        iterator = iter(dataloader)

        try:
            for _ in range(num_iteration):
                next(iterator)
        except StopIteration:
            raise AssertionError("StopIteration raised before the last iteration.")

        try:
            next(iterator)
        except StopIteration:
            return

        raise AssertionError("StopIteration not raised after the last iteration.")

    def test_stop_iteration(self):
        self.run_in_parallel(
            1,
            TestOobleckDataloader.run_until_stop,
            TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEP,
        )
