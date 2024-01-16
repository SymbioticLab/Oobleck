from oobleck import ExecutionEngine
from oobleck_colossalai import HeterogeneousParallelPlugin

from tqdm import tqdm

from dataclasses import dataclass
import simple_parsing

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import (
    AutoConfig,
    PretrainedConfig,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from data_builder import GLUEDataBuilder


@dataclass
class ExampleArguments:
    model_name_or_path: str = "gpt2"
    global_batch_size: int = 32
    num_epoch: int = 3
    warmup_faction: float = 0.1


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    plugin = HeterogeneousParallelPlugin(
        tp_size=4,
        global_batch_size=args.global_batch_size,
        microbatch_size=1,
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
    )
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

    for epoch in range(args.num_epoch):
        with tqdm(
            range(total_steps),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not (engine.is_master or is_pp_last_stage),
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


if __name__ == "__main__":
    main()
