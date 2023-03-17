import numpy as np

import torch
from itertools import chain
from typing import Optional, Tuple, Dict, List, Any, Type
from transformers import AutoTokenizer, AutoImageProcessor
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from datasets import Dataset, load_dataset, load_metric

from oobleck.module.model import lang_models, image_models


class OobleckDataset:
    """
    Load datasets from Hugging Face Hub (https://huggingface.co/datasets)
    and do preprocessing.
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str] = None,
    ):
        # TODO: replace it with evaluate.load("accuracy")
        metric = load_metric("accuracy")

        if any(lang_model in model_name for lang_model in lang_models):
            self.tokenizer, self.dataset = OobleckDataset.create_language_dataset(
                model_name, dataset_path, dataset_name
            )

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

            self.compute_metrics = compute_metrics

            self.data_collator = default_data_collator
        elif any(image_model in model_name for image_model in image_models):
            self.tokenizer, self.dataset = OobleckDataset.create_image_dataset(
                model_name, dataset_path, dataset_name
            )

            def compute_metrics(p):
                return metric.compute(
                    predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
                )

            self.compute_metrics = compute_metrics

            def collate_fn(examples):
                pixel_values = torch.stack(
                    [example["pixel_values"] for example in examples]
                )
                labels = torch.tensor([example["labels"] for example in examples])
                return {"pixel_values": pixel_values, "labels": labels}

            self.data_collator = collate_fn

        else:
            self.dataset = None

        assert (
            self.dataset
        ), f"Dataset it not initialized because given model {model_name} is not supported yet."

        trace_input = next(iter(self.dataset["train"]))
        self.trace_input_names = list(self.data_collator([trace_input]).keys())

    @staticmethod
    def create_image_dataset(
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str],
    ) -> Tuple[Type[BaseImageProcessor], Dataset]:
        dataset = load_dataset(dataset_path, dataset_name, task="image-classification")

        # If we don't have a validation split, split off a percentage of train as validation.
        if "validation" not in dataset.keys():
            split = dataset["train"].train_test_split(0.05)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]

        image_processor = AutoImageProcessor.from_pretrained(model_name)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )

        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )
        _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        def train_transforms(example_batch):
            """Apply _train_transforms across a batch."""
            example_batch["pixel_values"] = [
                _train_transforms(pil_img.convert("RGB"))
                for pil_img in example_batch["image"]
            ]
            return example_batch

        def val_transforms(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [
                _val_transforms(pil_img.convert("RGB"))
                for pil_img in example_batch["image"]
            ]
            return example_batch

        dataset["train"].set_transform(train_transforms)
        dataset["validation"].set_transform(val_transforms)

        return image_processor, dataset

    @staticmethod
    def create_language_dataset(
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str],
    ) -> Tuple[Type[PreTrainedTokenizer], Dataset]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        raw_dataset = load_dataset(dataset_path, dataset_name)
        if "validation" not in raw_dataset.keys():
            raw_dataset["validation"] = load_dataset(
                dataset_path,
                dataset_name,
                split=f"train[:5%]",
            )

        column_names = list(raw_dataset["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts, batched=True, load_from_cache_file=False
        )

        return tokenizer, tokenized_datasets
