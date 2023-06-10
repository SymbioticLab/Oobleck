import pytest
import torch
import torch.distributed

from oobleck.execution.pipeline import OobleckPipeline

from oobleck.csrc.planning.pipeline_template import (
    PipelineTemplateGenerator,
)

from transformers import TrainingArguments

import deepspeed.comm as dist
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR


@pytest.fixture
def singlegpu_pipeline_template(gpt2_model):
    # Must based on a real model
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        model_name=gpt2_model.model_name,
        model_tag=gpt2_model.model_tag,
        microbatch_size=1,
        num_nodes=(1, 1),
        num_gpus_per_node=1,
    )
    assert len(pipeline_templates) == 1
    return pipeline_templates[0]


@pytest.fixture
def twonodes_pipeline_template(gpt2_model):
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        model_name=gpt2_model.model_name,
        model_tag=gpt2_model.model_tag,
        microbatch_size=1,
        num_nodes=(2, 2),
        num_gpus_per_node=1,
    )
    assert len(pipeline_templates) > 0
    return pipeline_templates[0]


def test_initialize_pipeline(
    gpt2_model,
    dataloaders,
    singlegpu_pipeline_template,
    distributed_conf_one,
    distributed,
):
    training_args = TrainingArguments(output_dir="/tmp/test_output")

    assert dist.is_initialized()

    pg = torch.distributed.new_group()
    pipeline = OobleckPipeline(
        pipeline_template=singlegpu_pipeline_template,
        model=gpt2_model,
        dataloader=dataloaders[0],
        step=0,
        ranks=[0],
        process_group=pg,
        training_args=training_args,
    )
    assert pipeline.prev_rank is None
    assert pipeline.next_rank is None

    # Because we only have one rank, it should execute all layers in the model
    assert len(pipeline.model_layers) == len(gpt2_model.model)

    assert isinstance(pipeline.execution.optimizer, FusedAdam)
    assert isinstance(pipeline.execution.lr_scheduler, WarmupLR)
    assert pipeline.global_steps == 0
