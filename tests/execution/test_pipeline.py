from collections import OrderedDict
from typing import List

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from transformers import TrainingArguments

from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.layer import Layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_initialize_pipeline_single_stage(
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 GPUs required")
def test_initialize_two_pipelines(
    model_dataloaders, dummy_pipeline_template, init_distributed
):
    assert False, "Not implemented yet"

    model, train_dataloader, eval_dataloader = model_dataloaders
    init_distributed(True)
    assert dist.get_world_size() >= 2
    training_args = TrainingArguments(output_dir="/tmp/test_output")

    pipeline_template = dummy_pipeline_template(num_gpus=1)
    ranks = [dist.get_rank()]
    pg = torch.distributed.new_group(ranks)
    pipeline = OobleckPipeline(
        pipeline_template=pipeline_template,
        model=model,
        dataloader=train_dataloader,
        step=0,
        ranks=ranks,
        process_group=pg,
        training_args=training_args,
    )
    assert pipeline is not None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="at least 2 GPUs required")
def test_initialize_pipeline_multiple_stages(
    model_dataloaders, dummy_pipeline_template, init_distributed
):
    assert False, "Not implemented yet"

    model, train_dataloader, eval_dataloader = model_dataloaders
    init_distributed(True)
    assert dist.get_world_size() >= 2
    training_args = TrainingArguments(output_dir="/tmp/test_output")

    pipeline_template = dummy_pipeline_template(num_gpus=2)

    pg = torch.distributed.new_group()
    pipeline = OobleckPipeline(
        pipeline_template=pipeline_template,
        model=model,
        dataloader=train_dataloader,
        step=0,
        ranks=[0, 1],
        process_group=pg,
        training_args=training_args,
    )
    assert pipeline is not None


class TestOneStagePipeline:
    @pytest.fixture(autouse=True)
    def setup_method_fixture(
        self, dummy_pipeline: OobleckPipeline, request: pytest.FixtureRequest
    ):
        if not torch.cuda.is_available():
            pytest.skip("GPU required")

        self.num_gpus_per_pipeline: int = 1
        self.pipeline: OobleckPipeline = dummy_pipeline(
            num_gpus_per_pipeline=self.num_gpus_per_pipeline, ranks=[0]
        )[0]

    def test_load_microbatch(self):
        from tests.conftest import TRAIN_BATCH_SIZE

        assert self.pipeline.pipe_buffers["inputs"][0] is None
        self.pipeline.execution.load_microbatch(buffer_id=0)
        assert isinstance(self.pipeline.pipe_buffers["inputs"][0], tuple)
        assert all(
            isinstance(tensor, torch.Tensor)
            for tensor in self.pipeline.pipe_buffers["inputs"][0]
        )
        # Check batch size is correct
        assert all(
            tensor.shape[0] == TRAIN_BATCH_SIZE
            for tensor in self.pipeline.pipe_buffers["inputs"][0]
        )

    def test_forward(self):
        self.pipeline.execution.load_microbatch(buffer_id=0)
        assert self.pipeline.pipe_buffers["inputs"][0] is not None

        assert self.pipeline.pipe_buffers["outputs"][0] is None
        assert self.pipeline.execution.loss is None
        assert self.pipeline.execution.total_loss is None
        self.pipeline.execution.forward_pass(buffer_id=0)
        # because it is the last stage, output should still be None
        # Instead, it should write loss and total_loss
        assert self.pipeline.execution.loss is not None
        assert self.pipeline.execution.total_loss is not None

    def test_backward(self):
        self.pipeline.execution.load_microbatch(buffer_id=0)
        self.pipeline.execution.forward_pass(buffer_id=0)

        # backward_pass must clear outputs. Injecting a dummy value
        self.pipeline.pipe_buffers["outputs"][0] = torch.zeros(1)

        assert self.pipeline.execution.loss is not None
        assert self.pipeline.execution.total_loss is not None

        # before backward pass, check grad are none
        assert all(
            all(p.grad is None for p in layer.parameters())
            for layer in self.pipeline.model_layers
        )

        self.pipeline.execution.backward_pass(buffer_id=0)

        # check cleared by backward_pass
        assert self.pipeline.pipe_buffers["outputs"][0] is None

        # check gradients are generated by backward_pass
        assert all(
            all(
                p._grad is not None if p.requires_grad else None
                for p in layer.parameters()
            )
            for layer in self.pipeline.model_layers
        )

    def test_model_update(self):
        self.pipeline.execution.load_microbatch(buffer_id=0)
        self.pipeline.execution.forward_pass(buffer_id=0)
        self.pipeline.execution.backward_pass(buffer_id=0)

        # store parameters to compare later
        params = [
            [p.clone() for p in layer.parameters()]
            for layer in self.pipeline.model_layers
        ]

        self.pipeline.execution.optimizer_step()

        # model weights must have changed
        assert all(
            all(
                torch.any(torch.ne(p1.data, p2.data))
                for p1, p2 in zip(layer.parameters(), params[index])
            )
            for index, layer in enumerate(self.pipeline.model_layers)
        )

class TestTwoStagesPipeline:
    @pytest.fixture(autouse=True)
    def setup_method_fixture(
        self, dummy_pipeline: OobleckPipeline, request: pytest.FixtureRequest
    ):
        if not torch.cuda.device_count() >= 2:
            pytest.skip("at least 2 GPUs required")

        self.num_gpus_per_pipeline: int = 1
        self.pipeline: OobleckPipeline = dummy_pipeline(
            num_gpus_per_pipeline=self.num_gpus_per_pipeline, ranks=[0, 1]
        )[0]

    