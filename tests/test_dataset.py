from oobleck.execution.dataset import OobleckDataset
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor


def test_init_text_dataset():
    dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
    assert "train" in dataset.dataset
    assert "validation" in dataset.dataset
    assert isinstance(dataset.dataset["train"], Dataset)
    assert isinstance(dataset.dataset["validation"], Dataset)
    assert isinstance(dataset.tokenizer, PreTrainedTokenizerBase)


def test_init_image_dataset():
    dataset = OobleckDataset("microsoft/resnet-152", "Maysee/tiny-imagenet")
    assert "train" in dataset.dataset
    assert "validation" in dataset.dataset
    assert isinstance(dataset.dataset["train"], Dataset)
    assert isinstance(dataset.dataset["validation"], Dataset)
    assert isinstance(dataset.tokenizer, BaseImageProcessor)


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
