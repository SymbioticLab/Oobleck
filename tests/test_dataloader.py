import torch
import pytest
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.dataloader import OobleckDataLoader, LoaderType

from transformers import TrainingArguments
from typing import List

from .test_dataset import wikitext_dataset

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@pytest.fixture(scope="module")
def dataloaders(wikitext_dataset):
    training_args = TrainingArguments(
        output_dir="/tmp/output",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
    )
    training_dataloader = OobleckDataLoader(
        wikitext_dataset,
        training_args,
        LoaderType.Training,
        # total number of microbatches.
        # Currently only have one process, so it should be the same as
        # gradient_accumulation_steps.
        training_args.gradient_accumulation_steps,
        0,
        0,
    )
    eval_dataloader = OobleckDataLoader(
        wikitext_dataset,
        training_args,
        LoaderType.Evaluation,
        # total number of microbatches.
        # Currently only have one process, so it should be the same as
        # gradient_accumulation_steps.
        training_args.gradient_accumulation_steps,
        0,
        0,
    )
    return training_dataloader, eval_dataloader


@pytest.mark.dependency()
def test_dataloaders(dataloaders: List[OobleckDataLoader]):
    assert isinstance(dataloaders[0], OobleckDataLoader)
    assert isinstance(dataloaders[1], OobleckDataLoader)


@pytest.mark.dependency()
def test_samplers(dataloaders: List[OobleckDataLoader]):
    # Train dataloader microbatch size should be 8,
    # while eval dataloader microbatch size 4.
    assert dataloaders[0].batch_sampler.microbatch_size == TRAIN_BATCH_SIZE
    assert dataloaders[1].batch_sampler.microbatch_size == EVAL_BATCH_SIZE

    # Gradient accumulation must both be 2.
    assert dataloaders[0].batch_sampler.num_microbatches == GRADIENT_ACCUMULATION_STEP
    assert dataloaders[1].batch_sampler.num_microbatches == GRADIENT_ACCUMULATION_STEP

    # test consumed batch sample is 0.
    assert dataloaders[0].batch_sampler.consumed_samples == 0
    assert dataloaders[1].batch_sampler.consumed_samples == 0


@pytest.mark.dependency(depends=["test_samplers"])
def test_batch_train_samples(dataloaders):
    assert dataloaders[0].batch_sampler.consumed_samples == 0
    train_inputs = next(iter(dataloaders[0]))
    assert dataloaders[0].batch_sampler.consumed_samples == TRAIN_BATCH_SIZE
    assert isinstance(train_inputs, dict)
    for tensor in train_inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.size(dim=0) == TRAIN_BATCH_SIZE


@pytest.mark.dependency(depends=["test_samplers"])
def test_batch_eval_samples(dataloaders):
    assert dataloaders[1].batch_sampler.consumed_samples == 0
    eval_inputs = next(iter(dataloaders[1]))
    assert dataloaders[1].batch_sampler.consumed_samples == EVAL_BATCH_SIZE
    assert isinstance(eval_inputs, dict)
    for tensor in eval_inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.size(dim=0) == EVAL_BATCH_SIZE


@pytest.mark.dependency(depends=["test_batch_train_samples", "test_batch_eval_samples"])
def test_stop_iteration(dataloaders):
    num_iteration = len(dataloaders[0]) * GRADIENT_ACCUMULATION_STEP
    iterator = iter(dataloaders[0])

    try:
        for _ in range(num_iteration):
            next(iterator)
    except StopIteration:
        assert False, "StopIteration raised before the last iteration."

    with pytest.raises(StopIteration):
        next(iterator)
