import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from transformers import TrainingArguments

from oobleck.execution.pipeline import OobleckPipeline


def test_initialize_pipeline_single_template(
    model, dataloaders, dummy_pipeline_template, init_distributed
):
    init_distributed(True)
    training_args = TrainingArguments(output_dir="/tmp/test_output")

    assert dist.is_initialized()

    pipeline_template = dummy_pipeline_template(num_gpus=1)

    pg = torch.distributed.new_group()
    pipeline = OobleckPipeline(
        pipeline_template=pipeline_template,
        model=model,
        dataloader=dataloaders[0],
        step=0,
        ranks=[0],
        process_group=pg,
        training_args=training_args,
    )
    assert pipeline.prev_rank is None
    assert pipeline.next_rank is None

    # Because we only have one rank, it should execute all layers in the model
    assert len(pipeline.model_layers) == len(model.model)

    assert isinstance(pipeline.execution.optimizer, FusedAdam)
    assert isinstance(pipeline.execution.lr_scheduler, WarmupLR)
    assert pipeline.global_steps == 0
