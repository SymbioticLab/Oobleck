from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from types import MethodType
from typing import Any

import torch
import torch.distributed
import torch.fx
from deepspeed import comm as dist
from deepspeed.runtime.lr_schedules import WarmupLR
from deepspeed.runtime.pipe import schedule
from torch.distributed import ProcessGroup, Work
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import AdamW
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.execution.dataloader import OobleckDataLoader, OobleckSampler
from oobleck.execution.utils import DTYPE_TO_ID, ID_TO_DTYPE, zero_grads
from oobleck.module.model import OobleckModel


class PipelineExecution:
    """
    Pipeline execution module that this rank will use for training.
    For a single stage where this rank is in, there might be several ranks in FSDP group.

    TODO: explain shard_id. Heterogeneous pipeline could have different number of GPUs for the same layer.
    """

    def __init__(
        self,
        pipeline: OobleckPipeline,
        layers: list[FullyShardedDataParallel],
        shard_id: int,
        dataloader: OobleckDataLoader,
        training_args: TrainingArguments,
    ):
        self._pipeline = pipeline
        self._layers = layers
        self._shard_id = shard_id
        self._dataloader = dataloader
        self._data_iterator = iter(self._dataloader)
        self._training_args = training_args

        # stores the loss for the current microbatch being processed
        self._loss: torch.Tensor | Iterable[torch.Tensor] | None = None

        # stores the loss for the entire batch
        self.total_loss: torch.Tensor | Iterable[torch.Tensor] | None = None

        # TODO: use HF arguments to initialize optimizer and LR properly
        parameters = list(
            itertools.chain(*[list(layer.parameters()) for layer in self._layers])
        )
        self._optimizer = AdamW(
            parameters,
            lr=self._training_args.learning_rate,
            betas=(self._training_args.adam_beta1, self._training_args.adam_beta2),
            eps=self._training_args.adam_epsilon,
            fused=True,
        )
        num_training_steps = len(self._dataloader)
        self._lr_scheduler = WarmupLR(
            self.optimizer, self._training_args.get_warmup_steps(num_training_steps)
        )

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2454
    def _prepare_input(self, data: torch.Tensor | Any) -> torch.Tensor | Any:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            data = data.clone().detach().to(self._pipeline.device)
            data.requires_grad = data.is_floating_point()
            return data
        return data

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2472
    def _prepare_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor | Any]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        return tuple(self._prepare_input(t) for _, t in inputs.items())

    def load_microbatch(self, buffer_id: int):
        assert (
            self._pipeline.is_first_stage() or self._pipeline.is_last_stage()
        ), "load_microatch can only be called at either the first stage or the last stage."

        if self._pipeline.is_first_stage():
            batch = next(self._data_iterator)
            self._pipeline.pipe_buffers["inputs"][buffer_id] = self._prepare_inputs(
                batch
            )

    def forward_pass(self, buffer_id: int):
        inputs: tuple[torch.Tensor, ...] = self._pipeline.pipe_buffers["inputs"][
            buffer_id
        ]
        zero_grads(inputs)

        # XXX Hack
        # Some tensor might be converted from torch.Size().
        # Convert it to torch.Size so that forward can be executed
        inputs: tuple[torch.Size | torch.Tensor] = tuple(
            [
                torch.Size(input.tolist())
                if input.dim() == 1
                and input.data[0] == self._training_args.per_device_train_batch_size
                else input
                for input in inputs
            ]
        )

        # Execute forward
        for layer in self._layers:
            inputs = layer(*inputs)

        outputs = inputs

        # Optionally compute loss on the last stage
        if self._pipeline.is_last_stage():
            self._loss = outputs["loss"]
            del outputs["logits"]

            if isinstance(self._loss, torch.Tensor):
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self._loss)
                self.total_loss += self._loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self._loss]
                for idx, l in enumerate(self._loss):
                    assert torch.is_tensor(l)
                    self.total_loss[idx] += l.detach()
        else:
            # XXX Hack
            # It might includes torch.Size() in outputs.
            # Convert it to torch.Tensor so that it can be transferred
            outputs: tuple[torch.Tensor] = tuple(
                [
                    output
                    if torch.is_tensor(output)
                    else torch.LongTensor(data=output).to(self._device)
                    for output in outputs
                ]
            )

            self._pipeline.pipe_buffers["outputs"][buffer_id] = outputs

    def backward_pass(self, buffer_id: int):
        if self._pipeline.is_last_stage():
            loss = self._loss
            loss.backward()
        else:
            output_tensors: tuple[torch.Tensor] = self._pipeline.pipe_buffers[
                "outputs"
            ][buffer_id]
            output_tensors = tuple([t for t in output_tensors if t.requires_grad])
            grad_tensors: tuple[
                torch.Tensor
            ] = self._pipeline.communication.grad_recv_buf

            # Oobleck sharded model always returns tuple with tensors and torch.Size.
            assert len(output_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=output_tensors, grad_tensors=grad_tensors)

        # Free up memory from the output of forward()
        self._pipeline.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None

    def optimizer_step(self, lr_kwargs=None):
        # amp enable check: gradient clipping
        self._optimizer.step()
        self._lr_scheduler.step(**(lr_kwargs or {}))


class PipelineCommunication:
    def __init__(
        self,
        pipeline: OobleckPipeline,
        process_group: ProcessGroup,
        prev_rank: int | None,
        next_rank: int | None,
    ):
        self._pipeline = pipeline
        self._process_group = process_group
        self.prev_rank = prev_rank
        self.next_rank = next_rank

        self.sent_activation_meta: bool = False
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_activations`.
        self.activation_recv_buf: tuple[torch.Tensor] | None = None
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_gradients`.
        self.grad_recv_buf: tuple[torch.Tensor] | None = None

    def _send(
        self, tensor: torch.Tensor, dest_rank: int, async_op: bool = False
    ) -> Work:
        return (
            dist.isend(tensor, dest_rank, self._process_group)
            if async_op
            else dist.send(tensor, dest_rank, self._process_group)
        )

    def _recv(
        self, tensor: torch.Tensor, src_rank: int, async_op: bool = False
    ) -> Work:
        return (
            dist.irecv(tensor, src_rank, self._process_group)
            if async_op
            else dist.recv(tensor, src_rank, self._process_group)
        )

    def send_activations(self, buffer_id: int):
        def _send_activation_meta(buffer: tuple[torch.Tensor], receiver_rank: int):
            """Send activation dimension first to the next stage
            so that it can initialize buffers.

            Metadata is communicated in this order:
                * num_tensors in tensor tuple
                foreeach tensor in buffer:
                    * ndims
                    * dtype
                    * shape
                    * requires_grad
            """
            assert isinstance(
                buffer, tuple
            ), f"Could not send meta type {type(buffer)}."
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(
                self._pipeline.device
            )
            self._send(count_tensor, receiver_rank)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(
                    self._pipeline.device
                )
                send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(
                    self._pipeline.device
                )
                send_shape = torch.LongTensor(data=tensor.size()).to(
                    self._pipeline.device
                )
                send_req_grad = torch.LongTensor(
                    data=[1 if tensor.requires_grad else 0]
                ).to(self._pipeline.device)
                self._send(send_ndims, receiver_rank)
                self._send(send_dtype, receiver_rank)
                self._send(send_shape, receiver_rank)
                self._send(send_req_grad, receiver_rank)

        outputs: tuple[torch.Tensor] = self._pipeline.pipe_buffers["outputs"][buffer_id]
        if not self.sent_activation_meta:
            _send_activation_meta(outputs, self.next_rank)
            self.sent_activation_meta = True

        assert isinstance(outputs, tuple)
        for buffer in outputs:
            assert isinstance(buffer, torch.Tensor)
            self._send(buffer, self.next_rank)

    def recv_activations(self, buffer_id: int):
        def create_receive_buffer(sender_rank: int) -> tuple[torch.Tensor]:
            """Receive metadata about upcoming p2p transfers and return allocated buffer.

            Metadata is communicated in this order:
                * num_tensors in tensor tuple
                foreeach tensor in buffer:
                    * ndims
                    * dtype
                    * shape
                    * requires_grad
            """
            count_tensor = torch.LongTensor(data=[0]).to(self._pipeline.device)
            self._recv(count_tensor, sender_rank)
            num_tensors = count_tensor.item()
            buffers: list[torch.Tensor] = []
            for _ in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self._pipeline.device)
                self._recv(recv_ndims, sender_rank)
                recv_ndims = recv_ndims.item()

                recv_dtype = torch.LongTensor(data=[0]).to(self._pipeline.device)
                self._recv(recv_dtype, sender_rank)
                recv_dtype = ID_TO_DTYPE[recv_dtype.item()]

                recv_shape = torch.LongTensor([1] * recv_ndims).to(
                    self._pipeline.device
                )
                self._recv(recv_shape, sender_rank)
                recv_shape = recv_shape.tolist()

                recv_req_grad = torch.LongTensor(data=[0]).to(self._pipeline.device)
                self._recv(recv_req_grad, sender_rank)
                recv_req_grad = True if recv_req_grad.item() == 1 else False

                buffers.append(
                    torch.zeros(
                        recv_shape,
                        device=self._pipeline.device,
                        dtype=recv_dtype,
                        requires_grad=recv_req_grad,
                    )
                )
            return tuple(buffers)

        if self.activation_recv_buf is None:
            self.activation_recv_buf = create_receive_buffer(self.prev_rank)

        assert isinstance(self.activation_recv_buf, tuple)
        recvd: list[torch.Tensor | None] = [None] * len(self.activation_recv_buf)
        for idx, buffer in enumerate(self.activation_recv_buf):
            assert torch.is_tensor(buffer)
            self._recv(buffer, self.prev_rank)
            recvd[idx] = buffer.clone().detach()
            recvd[idx].requires_grad = buffer.requires_grad

        self._pipeline.pipe_buffers["inputs"][buffer_id] = tuple(recvd)

    def send_gradients(self, buffer_id: int):
        inputs = self._pipeline.pipe_buffers["inputs"][buffer_id]
        assert isinstance(inputs, tuple)

        for buffer in inputs:
            # Skip tensors that will not produce a gradient
            if not buffer.requires_grad:
                assert buffer.grad is None
                continue
            assert buffer.grad is not None
            self._send(buffer.grad, self.prev_rank)

        # We can free up the input buffer now
        self._pipeline.pipe_buffers["inputs"][buffer_id] = None

    def recv_gradients(self, buffer_id: int):
        def create_gradients_buffer(
            tensors: tuple[torch.Tensor],
        ) -> tuple[torch.Tensor]:
            assert isinstance(tensors, tuple)
            buffers: list[torch.Tensor] = []
            for tensor in tensors:
                assert isinstance(tensor, torch.Tensor)
                if tensor.requires_grad:
                    buffers.append(torch.zeros_like(tensor))

            return tuple(buffers)

        outputs = self._pipeline.pipe_buffers["outputs"][buffer_id]
        assert isinstance(outputs, tuple)

        # Allocate gradients if necessary
        if self.grad_recv_buf is None:
            self.grad_recv_buf = create_gradients_buffer(outputs)

        for buffer in self.grad_recv_buf:
            self._recv(buffer, self.next_rank)


# A map of PipeInstruction types to methods. Each method will be executed with the
# kwargs provided to the PipeInstruction from the scheduler.
INSTRUCTION_MAP = {
    schedule.OptimizerStep: PipelineExecution.optimizer_step,
    schedule.LoadMicroBatch: PipelineExecution.load_microbatch,
    schedule.ForwardPass: PipelineExecution.forward_pass,
    schedule.BackwardPass: PipelineExecution.backward_pass,
    schedule.SendActivation: PipelineCommunication.send_activations,
    schedule.RecvActivation: PipelineCommunication.recv_activations,
    schedule.SendGrad: PipelineCommunication.send_gradients,
    schedule.RecvGrad: PipelineCommunication.recv_gradients,
}


class OobleckPipeline:
    def __init__(
        self,
        pipeline_id: int,
        pipeline_template: PipelineTemplate,
        ranks: list[int],
        dataloader: OobleckDataLoader,
        step: int,
        training_args: TrainingArguments,
    ):
        self._pipeline_id = pipeline_id
        self._template = pipeline_template
        self._ranks = ranks
        self._dataloader = dataloader
        self._global_step = step
        self._training_args = training_args
        self.device = torch.device("cuda")

        assert dist.is_initialized(), "torch.distributed is not intialized."

        # This is used to indicate if we use this `OobleckPipeline` for training.
        self._my_pipeline = bool(dist.get_rank() in ranks)

        # Construct a 2D rank grid for this pipeline.
        # First dimension is for layer index, second dimension is for rank.
        self._rank_grid: dict[int, list[int]] = {}
        for stage in pipeline_template.get_stages():
            stage_ranks = ranks[: stage._num_gpus]
            ranks = ranks[stage._num_gpus :]

            # If length of `stage_ranks` is less than num_gpus_per_node, adjust it
            # so that it conforms a full 2D grid
            if stage_ranks < pipeline_template._num_gpus_per_node:
                stage_ranks = list(
                    itertools.repeat(
                        rank, pipeline_template._num_gpus_per_node // stage_ranks
                    )
                    for rank in stage_ranks
                )

            for layer_index in stage._layer_indices:
                self._rank_grid[layer_index] = stage_ranks

        assert len(ranks) == 0, "Not all ranks were assigned to a stage."

    def train(self):
        for step_cmds in self.train_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                _exec_instr = MethodType(INSTRUCTION_MAP[type(cmd)], self)
                _exec_instr(**cmd.kwargs)

        self._global_step += 1

    def get_rank_for_id(self, layer_id: int, shard_id: int) -> int:
        return self._rank_grid[layer_id][shard_id]

    def initialize_execution(
        self,
        layers: list[FullyShardedDataParallel],
        shard_id: int,
    ):
        self.execution = PipelineExecution(
            pipeline=self,
            layers=layers,
            shard_id=shard_id,
            dataloader=self._dataloader,
        )

        # initialize_execution assumes to be called only if this rank is involved in
        # the pipeline. Failure of getting my_layer_index cannot happen.
        my_rank = dist.get_rank()
        my_layer_index = next(my_rank in ranks for ranks in self._rank_grid.values())
        my_stage_index = next(
            stage_index
            for stage_index, stage in enumerate(self._template.get_stages())
            if my_layer_index in stage._layer_indices
        )

        sampler: OobleckSampler = self._dataloader.batch_sampler
        self.train_schedule = schedule.TrainSchedule(
            micro_batches=sampler.num_microbatches[self._pipeline_id],
            num_stages=len(self._template.get_stages()),
            stage_id=my_stage_index,
        )

        num_pipe_buffers = self.train_schedule.num_pipe_buffers()
        self.pipe_buffers: dict[str, list[tuple[torch.Tensor] | None]] = {
            # batch input and received activations
            "inputs": [None for _ in range(num_pipe_buffers)],
            # labels from batch input
            "labels": [None for _ in range(num_pipe_buffers)],
            # activations to be sent
            "outputs": [None for _ in range(num_pipe_buffers)],
        }

    def initialize_distributed_fsdp(self, model: OobleckModel):
        """Initialize torch.distributed.process_groups per layer.
        Even I am not involved in a group, torch.distributed requires all ranks to call
        `new_group()`. Thus this method should be called by everyone.

        Plus, if this rank is involved in a group, initialize execution.
        """
        self._per_layer_pgs: dict[int, ProcessGroup] = {}
        self.execution: PipelineExecution | None = None

        fsdp_layers: list[FullyShardedDataParallel] = []
        shard_id: int = -1
        my_rank = dist.get_rank()
        for layer_id, ranks in self._rank_grid.items():
            # Remove potential duplicates
            pg = dist.new_group(list(set(ranks)))
            self._per_layer_pgs[layer_id] = pg

            # Get FSDP module if this rank is involved in this layer
            if my_rank in ranks:
                fsdp_layers.append(
                    FullyShardedDataParallel(
                        model.model[layer_id], process_group=pg, device_id=self.device
                    )
                )
                shard_id = ranks.index(my_rank)

        if fsdp_layers:
            assert shard_id >= 0, "shard id is not set while fsdp_layers have layers."
            self.initialize_execution(fsdp_layers, shard_id)

        assert len(self._per_layer_pgs) == len(
            model.model
        ), "Number of per-layer process groups and model layers must match."
        assert all(
            layer_index in self._per_layer_pgs
            for layer_index in range(len(model.model))
        ), "Process groups for some layers are not initialized."

        # self.execution may not be initialized at this moment. Don't add assertion here.

    def initialize_distributed_pipeline(self):
        """Initialize torch.distributed.process_groups for a FSDP sharded pipeline.
        Even I am not involved in a group, torch.distributed requires all ranks to call
        `new_group()`. Thus this method should be called by everyone.

        Plus, if this rank is involved in a group, initialize communication.
        """
        self._per_sharded_pp_pgs: dict[int, ProcessGroup] = {}
        self.communication: PipelineCommunication | None = None

        my_rank = dist.get_rank()
        for shard_id in range(len(self._rank_grid[0])):
            ranks: list[int] = [
                ranks_per_layer[shard_id] for ranks_per_layer in self._rank_grid
            ]
            # Remove potential duplicates
            pg = dist.new_group(list(set(ranks)))
            self._per_sharded_pp_pgs[shard_id] = pg

            if my_rank in ranks:
                rank_index = ranks.index(my_rank)
                self.communication = PipelineCommunication(
                    pipeline=self,
                    process_group=pg,
                    prev_rank=ranks[rank_index - 1] if rank_index > 0 else None,
                    next_rank=ranks[rank_index + 1]
                    if rank_index < len(ranks) - 1
                    else None,
                )

        assert len(self._per_sharded_pp_pgs) == len(
            self._rank_grid[0]
        ), "Number of per-shard process groups and model layers must match."

        # self.communication may not be initialized at this moment. Don't add assertion here.

    def is_first_stage(self) -> bool:
        return self.communication.prev_rank is None

    def is_last_stage(self) -> bool:
        return self.communication.next_rank is None
