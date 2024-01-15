from oobleck import ExecutionEngine
from oobleck_colossalai import HeterogeneousParallelPlugin

from tqdm import tqdm

from dataclasses import dataclass
import datasets
import simple_parsing

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
    PretrainedConfig,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)


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


@dataclass
class ExampleArguments:
    model_name_or_path: str = "gpt2"
    global_batch_size: int = 32
    num_epoch: int = 3
    warmup_faction: float = 0.1


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    plugin = HeterogeneousParallelPlugin(tp_size=2, microbatch_size=1, precision="fp32")
    engine = ExecutionEngine(plugin)

    config: PretrainedConfig = AutoConfig.from_pretrained(args.model_name_or_path)
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    # Prepare dataloader
    data_builder = GLUEDataBuilder(args.model_name_or_path, plugin, task_name="mrpc")
    dataloader = data_builder.dataloader()

    # optimizer
    optimizer = Adam(model.parameters())

    # lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(total_steps * args.warmup_faction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _criterion, _, lr_scheduler = engine.prepare(
        model,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Train model
    model.train()
    optimizer.zero_grad()
    dataloader_iter = iter(dataloader)

    is_pp_last_stage = engine.plugin.stage_manager.is_last_stage()

    for epoch in args.num_epoch:
        with tqdm(
            range(total_steps),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not (engine.is_master() or is_pp_last_stage),
        ) as pbar:
            for _ in pbar:
                outputs = engine.execute(
                    dataloader_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs.loss,
                    optimizer=optimizer,
                    return_loss=True,
                    return_outputs=True,
                )

                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
