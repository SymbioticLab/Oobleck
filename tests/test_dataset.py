from oobleck.execution.dataset import OobleckDataset
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

import pytest


@pytest.fixture(scope="session")
def wikitext_dataset():
    return OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")


@pytest.fixture(scope="session")
def imagenet_dataset():
    return OobleckDataset("microsoft/resnet-152", "Maysee/tiny-imagenet")


def test_init_text_dataset(wikitext_dataset):
    assert "train" in wikitext_dataset.dataset
    assert "validation" in wikitext_dataset.dataset
    assert isinstance(wikitext_dataset.dataset["train"], Dataset)
    assert isinstance(wikitext_dataset.dataset["validation"], Dataset)
    assert isinstance(wikitext_dataset.tokenizer, PreTrainedTokenizerBase)


def test_init_image_dataset(imagenet_dataset):
    assert "train" in imagenet_dataset.dataset
    assert "validation" in imagenet_dataset.dataset
    assert isinstance(imagenet_dataset.dataset["train"], Dataset)
    assert isinstance(imagenet_dataset.dataset["validation"], Dataset)
    assert isinstance(imagenet_dataset.tokenizer, BaseImageProcessor)


# class TestOobleckDataset(unittest.TestCase):
#     def test_init_text_dataset(self):
#         dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
#         self.assertIn("train", dataset.dataset)
#         self.assertIn("validation", dataset.dataset)
#         self.assertIsInstance(dataset.dataset["train"], Dataset)
#         self.assertIsInstance(dataset.dataset["validation"], Dataset)
#         self.assertIsInstance(dataset.tokenizer, PreTrainedTokenizerBase)

#     def test_init_image_dataset(self):
#         dataset = OobleckDataset("microsoft/resnet-152", "Maysee/tiny-imagenet")
#         self.assertIn("train", dataset.dataset)
#         self.assertIn("validation", dataset.dataset)
#         self.assertIsInstance(dataset.dataset["train"], Dataset)
#         self.assertIsInstance(dataset.dataset["validation"], Dataset)
#         self.assertIsInstance(dataset.tokenizer, BaseImageProcessor)


# if __name__ == "__main__":
#     unittest.main()
