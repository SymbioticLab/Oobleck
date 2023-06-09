import pytest
import torch
import torch.distributed

from oobleck.execution.pipeline import OobleckPipeline

from oobleck.csrc.planning.pipeline_template import (
    PipelineTemplateGenerator,
)

from transformers import TrainingArguments


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
