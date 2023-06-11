import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from transformers import TrainingArguments

from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_initialize_pipeline_single_template(
    model_dataloaders, dummy_pipeline_template, init_distributed
):
    model, train_dataloader, eval_dataloader = model_dataloaders
    init_distributed(True)
    training_args = TrainingArguments(output_dir="/tmp/test_output")

    pipeline_template = dummy_pipeline_template(num_gpus=1)

    pg = torch.distributed.new_group()
    pipeline = OobleckPipeline(
        pipeline_template=pipeline_template,
        model=model,
        dataloader=train_dataloader,
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

    for layer in pipeline.model_layers:
        assert all(p.is_cuda for p in layer.parameters())


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
# def test_initialize_pipeline_multiple_templates(
#     model, dataloaders, dummy_pipeline_template, init_distributed
# ):
#     init_distributed(True)
#     training_args = TrainingArguments(output_dir="/tmp/test_output")
