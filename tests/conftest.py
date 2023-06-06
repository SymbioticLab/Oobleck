from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.dataloader import OobleckDataLoader, LoaderType
from oobleck.module.model import OobleckModel
from pipeline_template import PipelineTemplateGenerator

from transformers import TrainingArguments
import pytest

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@pytest.fixture(scope="session")
def wikitext_dataset():
    return OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")


@pytest.fixture(scope="session")
def imagenet_dataset():
    return OobleckDataset("microsoft/resnet-152", "Maysee/tiny-imagenet")


@pytest.fixture
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


@pytest.fixture(scope="session")
def gpt2_model(wikitext_dataset):
    # Refer to oobleck/examples/*.py for model arguments
    # gpt2-medium
    model_args = {
        "num_hidden_layers": 32,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    }
    return OobleckModel("gpt2", wikitext_dataset.sample, None, "test", model_args)


@pytest.fixture(scope="module")
def pipeline_template_generator():
    return PipelineTemplateGenerator()
