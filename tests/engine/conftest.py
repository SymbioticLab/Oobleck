import csv
from pathlib import Path

import datasets
import pytest
from oobleck_colossalai import HeterogeneousParallelPlugin
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer


def init_profile_data(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_index",
                "layer_name",
                "forward",
                "backward",
                "mem_required",
            ],
        )
        writer.writeheader()
        for index, layer_name in enumerate(
            ["transformer.wte", "transformer.wpe", "transformer.drop"]
            + [f"transformer.h.{i}" for i in range(0, 4)]
            + [f"transformer.ln_f", "score"]
        ):
            writer.writerow(
                {
                    "layer_index": index,
                    "layer_name": layer_name,
                    "forward": 1.0,
                    "backward": 1.0,
                    "mem_required": 10,
                }
            )


singlenode_template = {
    PipelineTemplate(
        modules_per_stage=[
            ["transformer.wte", "transformer.wpe", "transformer.drop"]
            + [f"transformer.h.{i}" for i in range(0, 4)]
            + [f"transformer.ln_f", "score"],
        ],
    ): 1
}

homogeneous_templates = {
    # 3 nodes
    PipelineTemplate(
        modules_per_stage=[
            [
                "transformer.wte",
                "transformer.wpe",
                "transformer.drop",
                "transformer.h.0",
            ],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "score"],
        ],
    ): 3
}
heterogeneous_templates = {
    # 3 nodes
    PipelineTemplate(
        modules_per_stage=[
            [
                "transformer.wte",
                "transformer.wpe",
                "transformer.drop",
                "transformer.h.0",
            ],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "score"],
        ],
    ): 1,
    # 2 nodes
    PipelineTemplate(
        modules_per_stage=[
            ["transformer.wte", "transformer.wpe", "transformer.drop"],
            [f"transformer.h.{i}" for i in range(0, 4)]
            + [f"transformer.ln_f", "score"],
        ],
    ): 3,
}


class GLUEDataBuilder:
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        plugin: HeterogeneousParallelPlugin,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
    ):
        self.plugin = plugin
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = datasets.load_dataset("glue", task_name)

        def convert_to_features(example_batch):
            text_fields = GLUEDataBuilder.task_text_field_map[task_name]
            if len(text_fields) > 1:
                texts_or_text_pairs = list(
                    zip(example_batch[text_fields[0]], example_batch[text_fields[1]])
                )
            else:
                texts_or_text_pairs = example_batch[text_fields[0]]

            # Tokenize the text/text pairs
            features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
            )

            features["labels"] = example_batch["label"]
            return features

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in GLUEDataBuilder.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def dataloader(self) -> DataLoader:
        return self.plugin.prepare_dataloader(
            self.dataset["train"],
            shuffle=True,
            drop_last=True,
        )


@pytest.fixture(scope="class", params=[homogeneous_templates, heterogeneous_templates])
def plugin(request: pytest.FixtureRequest):
    plugin = HeterogeneousParallelPlugin(tp_size=2, microbatch_size=1)
    plugin.set_pipeline_templates(request.param)
    request.cls.plugin = plugin
