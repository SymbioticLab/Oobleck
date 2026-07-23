import functools

import click
import datasets
from cornstarch.plugin.heterogeneous_dataloader import HeterogeneousDataLoader
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from oobleck.engine.execution_engine import ExecutionEngine
from oobleck.engine.plugin import OobleckPlugin


def tokenize_batch_for_pretrain(
    batch, tokenizer: PreTrainedTokenizer | None = None, max_length: int = 2048
):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


@click.command
@click.option(
    "--model_name_or_path", type=str, default="gpt2", help="Model name or path."
)
@click.option("--global_batch_size", type=int, default=96, help="Global batch size.")
@click.option("--num_epoch", type=int, default=3, help="Number of epochs.")
@click.option("--warmup_faction", type=float, default=0.1, help="Warmup faction.")
@click.option("--tp_size", type=int, default=1, help="Tensor parallel degree.")
def main(
    model_name_or_path: str,
    global_batch_size: int,
    num_epoch: int,
    warmup_faction: float,
    tp_size: int,
):
    plugin = OobleckPlugin(
        tp_size=tp_size,
        global_batch_size=global_batch_size,
        microbatch_size=2,
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
        fault_tolerance_threshold=1,
    )

    engine = ExecutionEngine(plugin)

    config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)
    model.gradient_checkpointing_enable()

    # Prepare dataloader
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    dataloader: HeterogeneousDataLoader = plugin.prepare_dataloader(
        dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(
            tokenize_batch_for_pretrain,
            tokenizer=tokenizer,
            max_length=model.config.max_position_embeddings,
        ),
    )

    # optimizer
    optimizer = Adam(model.parameters())

    # lr scheduler
    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_faction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _, dataloader, lr_scheduler = engine.prepare(
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

    for epoch in range(num_epoch):
        total_step = len(dataloader)
        dataloader_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
            disable=not (engine.is_master or is_pp_last_stage),
        ) as pbar:
            for _ in pbar:
                outputs = engine.execute(
                    dataloader_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs.loss,
                    optimizer=optimizer,
                    return_loss=True,
                    return_outputs=False,
                )

                if outputs is None:
                    # Reconfiguration due to failure is done.
                    model, optimizer, dataloader = engine.reconfigure(
                        model, optimizer, dataloader
                    )
                    logger.warning("Reconfiguration is done. Restarting training.")
                    dataloader_iter = iter(dataloader)
                    continue

                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()


if __name__ == "__main__":
    main()
