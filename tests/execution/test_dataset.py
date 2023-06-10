from oobleck.execution.dataset import OobleckDataset
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor


def test_init_text_dataset(wikitext_dataset):
    assert isinstance(wikitext_dataset, OobleckDataset)
    assert "train" in wikitext_dataset.dataset
    assert "validation" in wikitext_dataset.dataset
    assert isinstance(wikitext_dataset.dataset["train"], Dataset)
    assert isinstance(wikitext_dataset.dataset["validation"], Dataset)
    assert isinstance(wikitext_dataset.tokenizer, PreTrainedTokenizerBase)


def test_init_image_dataset(imagenet_dataset):
    assert isinstance(imagenet_dataset, OobleckDataset)
    assert "train" in imagenet_dataset.dataset
    assert "validation" in imagenet_dataset.dataset
    assert isinstance(imagenet_dataset.dataset["train"], Dataset)
    assert isinstance(imagenet_dataset.dataset["validation"], Dataset)
    assert isinstance(imagenet_dataset.tokenizer, BaseImageProcessor)
