import pytest
import torch
from transformers import TrainingArguments

from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from tests.conftest import (EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEP,
                            TRAIN_BATCH_SIZE)


@pytest.mark.parametrize("dataset", ["wikitext_dataset", "imagenet_dataset"])
@pytest.mark.parametrize("consumed_sample", [0, 40])
@pytest.mark.parametrize("type", [LoaderType.Training, LoaderType.Evaluation])
def test_initialize_dataloader(
    dataset, consumed_sample, type, request: pytest.FixtureRequest
):
    dataset = request.getfixturevalue(dataset)
    training_args = TrainingArguments(
        output_dir="/tmp/output",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
    )
    dataloader = OobleckDataLoader(
        datasets=dataset,
        args=training_args,
        dataloader_type=type,
        num_total_microbatches=TRAIN_BATCH_SIZE,
        consumed_samples=consumed_sample,
        epoch=0,
    )
    assert isinstance(dataloader, OobleckDataLoader)
    assert (
        dataloader.batch_sampler.microbatch_size == TRAIN_BATCH_SIZE
        if type == LoaderType.Training
        else EVAL_BATCH_SIZE
    )
    assert dataloader.batch_sampler.num_microbatches == GRADIENT_ACCUMULATION_STEP
    assert dataloader.batch_sampler.consumed_samples == consumed_sample


def test_batch_train_samples(dataloaders):
    assert dataloaders[0].batch_sampler.consumed_samples == 0
    train_inputs = next(iter(dataloaders[0]))
    assert dataloaders[0].batch_sampler.consumed_samples == TRAIN_BATCH_SIZE
    assert isinstance(train_inputs, dict)
    for tensor in train_inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.size(dim=0) == TRAIN_BATCH_SIZE


def test_batch_eval_samples(dataloaders):
    assert dataloaders[1].batch_sampler.consumed_samples == 0
    eval_inputs = next(iter(dataloaders[1]))
    assert dataloaders[1].batch_sampler.consumed_samples == EVAL_BATCH_SIZE
    assert isinstance(eval_inputs, dict)
    for tensor in eval_inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.size(dim=0) == EVAL_BATCH_SIZE


def test_next_batch(dataloaders):
    for dataloader in dataloaders:
        iterator = iter(dataloader)
        sample = next(iterator)
        assert isinstance(sample, dict)


def test_stop_iteration(dataloaders):
    for dataloader in dataloaders:
        num_iteration = len(dataloader) * GRADIENT_ACCUMULATION_STEP
        iterator = iter(dataloader)

        try:
            for _ in range(num_iteration):
                next(iterator)
        except StopIteration:
            assert False, "StopIteration raised before the last iteration."

        with pytest.raises(StopIteration):
            next(iterator)
