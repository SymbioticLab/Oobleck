from __future__ import annotations

import copy
import itertools
import weakref
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed
import torch.fx
from deepspeed import comm as dist
from deepspeed.runtime.lr_schedules import WarmupLR
from deepspeed.runtime.pipe import schedule
from torch.distributed import ProcessGroup, Work
from torch.optim import AdamW
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.execution.dataloader import OobleckDataLoader, OobleckSampler
from oobleck.execution.fsdp import FullyShardedDataParallelLayer, StreamType
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
        layers: list[FullyShardedDataParallelLayer],
        shard_id: int,
        dataloader: OobleckDataLoader,
        training_args: TrainingArguments,
    ):
        self._pipeline = weakref.ref(pipeline)
        self._layers = layers
        self._shard_id = shard_id
        self._dataloader = dataloader
        self._data_iterator = iter(self._dataloader)
        self._training_args = training_args

        # stores the loss for the current microbatch being processed
        self._loss: torch.Tensor | None = None

        # stores the loss for the entire batch
        self.total_loss: torch.Tensor | None = None

        # TODO: use HF arguments to initialize optimizer and LR properly
        parameters = list(l._param_handle.flat_param for l in layers)
        self._optimizer = AdamW(
            parameters,
            lr=self._training_args.learning_rate,
            betas=(self._training_args.adam_beta1, self._training_args.adam_beta2),
            eps=self._training_args.adam_epsilon,
            fused=True,
        )
        num_training_steps = len(self._dataloader)
        self._lr_scheduler = WarmupLR(
            self._optimizer, self._training_args.get_warmup_steps(num_training_steps)
        )

    @property
    def pipeline(self) -> OobleckPipeline:
        return self._pipeline()

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
            data = data.clone().detach().to(self.pipeline.device)
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
            self.pipeline.is_first_stage() or self.pipeline.is_last_stage()
        ), "load_microatch can only be called at either the first stage or the last stage."

        if self.pipeline.is_first_stage():
            batch = next(self._data_iterator)
            self.pipeline.pipe_buffers["inputs"][buffer_id] = self._prepare_inputs(
                batch
            )

    def forward_pass(self, buffer_id: int):
        inputs: tuple[torch.Tensor, ...] = self.pipeline.pipe_buffers["inputs"][
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
        if self.pipeline.is_last_stage():
            self._loss = outputs[0]

            assert isinstance(self._loss, torch.Tensor)
            if self.total_loss is None:
                self.total_loss = torch.zeros_like(self._loss)
            self.total_loss += self._loss.detach()

        else:
            # XXX Hack
            # It might includes torch.Size() in outputs.
            # Convert it to torch.Tensor so that it can be transferred
            outputs: tuple[torch.Tensor] = tuple(
                [
                    output
                    if torch.is_tensor(output)
                    else torch.LongTensor(data=output).to(self.pipeline.device)
                    for output in outputs
                ]
            )

            self.pipeline.pipe_buffers["outputs"][buffer_id] = outputs

    def backward_pass(self, buffer_id: int):
        if self.pipeline.is_last_stage():
            loss = self._loss
            loss.backward()
        else:
            output_tensors: tuple[torch.Tensor] = self.pipeline.pipe_buffers["outputs"][
                buffer_id
            ]
            output_tensors = tuple([t for t in output_tensors if t.requires_grad])
            grad_tensors: tuple[
                torch.Tensor
            ] = self.pipeline.communication.grad_recv_buf

            # Oobleck sharded model always returns tuple with tensors and torch.Size.
            assert len(output_tensors) == len(grad_tensors)

            torch.autograd.backward(tensors=output_tensors, grad_tensors=grad_tensors)

        # Free up memory from the output of forward()
        self.pipeline.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None
        self._loss = None

    def optimizer_step(self, lr_kwargs=None):
        # amp enable check: gradient clipping
        for l in self._layers:
            l._param_handle.prepare_gradient_for_optim()
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
        self._pipeline = weakref.ref(pipeline)
        self._process_group = process_group
        self.prev_rank = prev_rank
        self.next_rank = next_rank

        self.sent_activation_meta: bool = False
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_activations`.
        self.activation_recv_buf: tuple[torch.Tensor] | None = None
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_gradients`.
        self.grad_recv_buf: tuple[torch.Tensor] | None = None

    @property
    def pipeline(self) -> OobleckPipeline:
        return self._pipeline()

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
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.pipeline.device)
            self._send(count_tensor, receiver_rank)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(
                    self.pipeline.device
                )
                send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(
                    self.pipeline.device
                )
                send_shape = torch.LongTensor(data=tensor.size()).to(
                    self.pipeline.device
                )
                send_req_grad = torch.LongTensor(
                    data=[1 if tensor.requires_grad else 0]
                ).to(self.pipeline.device)
                self._send(send_ndims, receiver_rank)
                self._send(send_dtype, receiver_rank)
                self._send(send_shape, receiver_rank)
                self._send(send_req_grad, receiver_rank)

        outputs: tuple[torch.Tensor] = self.pipeline.pipe_buffers["outputs"][buffer_id]
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
            count_tensor = torch.LongTensor(data=[0]).to(self.pipeline.device)
            self._recv(count_tensor, sender_rank)
            num_tensors = count_tensor.item()
            buffers: list[torch.Tensor] = []
            for _ in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_ndims, sender_rank)
                recv_ndims = recv_ndims.item()

                recv_dtype = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_dtype, sender_rank)
                recv_dtype = ID_TO_DTYPE[recv_dtype.item()]

                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.pipeline.device)
                self._recv(recv_shape, sender_rank)
                recv_shape = recv_shape.tolist()

                recv_req_grad = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_req_grad, sender_rank)
                recv_req_grad = True if recv_req_grad.item() == 1 else False

                buffers.append(
                    torch.zeros(
                        recv_shape,
                        device=self.pipeline.device,
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

        self.pipeline.pipe_buffers["inputs"][buffer_id] = tuple(recvd)

    def send_gradients(self, buffer_id: int):
        inputs = self.pipeline.pipe_buffers["inputs"][buffer_id]
        assert isinstance(inputs, tuple)

        for buffer in inputs:
            # Skip tensors that will not produce a gradient
            if not buffer.requires_grad:
                assert buffer.grad is None
                continue
            assert buffer.grad is not None
            self._send(buffer.grad, self.prev_rank)

        # We can free up the input buffer now
        self.pipeline.pipe_buffers["inputs"][buffer_id] = None

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

        outputs = self.pipeline.pipe_buffers["outputs"][buffer_id]
        assert isinstance(outputs, tuple)

        # Allocate gradients if necessary
        if self.grad_recv_buf is None:
            self.grad_recv_buf = create_gradients_buffer(outputs)

        for buffer in self.grad_recv_buf:
            self._recv(buffer, self.next_rank)

    def reduce_gradients(self):
        pass

    def reduce_tied_gradients(self):
        pass


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
            if stage._num_gpus < pipeline_template._num_gpus_per_node:
                stage_ranks = [
                    list(
                        itertools.repeat(
                            rank,
                            pipeline_template._num_gpus_per_node // len(stage_ranks),
                        )
                    )
                    for rank in stage_ranks
                ]
                stage_ranks = list(itertools.chain.from_iterable(stage_ranks))

            for layer_index in stage._layer_indices:
                self._rank_grid[layer_index] = stage_ranks

        assert len(ranks) == 0, "Not all ranks were assigned to a stage."

    def train(self):
        # A map of PipeInstruction types to methods. Each method will be executed with the
        # kwargs provided to the PipeInstruction from the scheduler.
        instruction_map = {
            schedule.OptimizerStep: self.execution.optimizer_step,
            schedule.LoadMicroBatch: self.execution.load_microbatch,
            schedule.ForwardPass: self.execution.forward_pass,
            schedule.BackwardPass: self.execution.backward_pass,
            schedule.SendActivation: self.communication.send_activations,
            schedule.RecvActivation: self.communication.recv_activations,
            schedule.SendGrad: self.communication.send_gradients,
            schedule.RecvGrad: self.communication.recv_gradients,
            schedule.ReduceGrads: self.communication.reduce_gradients,
            schedule.ReduceTiedGrads: self.communication.reduce_tied_gradients,
        }

        for step_cmds in self.train_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in instruction_map:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self.[execution|communication].func(buffer_id)
                instruction_map[type(cmd)](**cmd.kwargs)

        # Cleanup buffers
        for name, pipe_buffers in self.pipe_buffers.items():
            self.pipe_buffers[name] = [None] * len(pipe_buffers)

        self._global_step += 1

    def get_rank_for_id(self, layer_id: int, shard_id: int) -> int:
        return self._rank_grid[layer_id][shard_id]

    def _initialize_execution(
        self,
        layers: list[FullyShardedDataParallelLayer],
        shard_id: int,
    ):
        self.execution = PipelineExecution(
            pipeline=self,
            layers=layers,
            shard_id=shard_id,
            dataloader=self._dataloader,
            training_args=self._training_args,
        )

        # initialize_execution assumes to be called only if this rank is involved in
        # the pipeline. Failure of getting my_layer_index cannot happen.
        my_rank = dist.get_rank()
        my_layer_index = next(
            layer_index
            for layer_index, ranks in self._rank_grid.items()
            if my_rank in ranks
        )
        my_stage_index = next(
            stage_index
            for stage_index, stage in enumerate(self._template.get_stages())
            if my_layer_index in stage._layer_indices
        )

        sampler: OobleckSampler = self._dataloader.batch_sampler
        self.train_schedule = schedule.TrainSchedule(
            micro_batches=sampler.num_microbatches[self._pipeline_id],
            stages=len(self._template.get_stages()),
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

        fsdp_layers: list[FullyShardedDataParallelLayer] = []
        shard_id: int = -1
        my_rank = dist.get_rank()
        for layer_id, ranks in self._rank_grid.items():
            # Remove potential duplicates
            pg = dist.new_group(list(set(ranks)))
            self._per_layer_pgs[layer_id] = pg

            unshard_stream = torch.cuda.Stream()
            # Get FSDP module if this rank is involved in this layer
            if my_rank in ranks:
                fsdp_layer = FullyShardedDataParallelLayer(
                    model.layers[layer_id].to("cuda"),
                    process_group=pg,
                    streams={
                        StreamType.UNSHARD: unshard_stream,
                    },
                )
                fsdp_layer._param_handle.init_flat_param_attributes()
                fsdp_layers.append(fsdp_layer)
                shard_id = ranks.index(my_rank)

        if fsdp_layers:
            assert shard_id >= 0, "shard id is not set while fsdp_layers have layers."
            self._initialize_execution(fsdp_layers, shard_id)

        assert len(self._per_layer_pgs) == len(
            model.layers
        ), "Number of per-layer process groups and model layers must match."
        assert all(
            layer_index in self._per_layer_pgs
            for layer_index in range(len(model.layers))
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
                ranks_per_layer[shard_id]
                for ranks_per_layer in self._rank_grid.values()
            ]
            # Remove potential duplicates
            pg = dist.new_group(list(set(ranks)))
            self._per_sharded_pp_pgs[shard_id] = pg

            if my_rank in ranks:
                unique_ranks = list(set(ranks))
                rank_index = unique_ranks.index(my_rank)
                self.communication = PipelineCommunication(
                    pipeline=self,
                    process_group=pg,
                    prev_rank=unique_ranks[rank_index - 1] if rank_index > 0 else None,
                    next_rank=unique_ranks[rank_index + 1]
                    if rank_index < len(unique_ranks) - 1
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
