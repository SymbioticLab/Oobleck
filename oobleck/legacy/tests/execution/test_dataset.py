from datasets import Dataset
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizerBase

from oobleck.execution.dataset import OobleckDataset
from tests.conftest import OobleckSingleProcessTestCase


class TestDataset(OobleckSingleProcessTestCase):
    def test_attributes_type(self):
        dataset = self.factory.get_dataset()
        assert isinstance(dataset, OobleckDataset)
        assert "train" in dataset.dataset
        assert "validation" in dataset.dataset
        assert isinstance(dataset.dataset["train"], Dataset)
        assert isinstance(dataset.dataset["validation"], Dataset)
        # assert isinstance(dataset.tokenizer, PreTrainedTokenizerBase)
