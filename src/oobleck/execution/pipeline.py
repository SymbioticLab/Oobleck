import torch
import torch.distributed
import itertools

from types import MethodType
from typing import Union, Any, Dict, Mapping, Tuple, List, Optional, Iterable

from torch.distributed import ProcessGroup, Work
from deepspeed import comm as dist
from deepspeed.utils import logger, instrument_w_nvtx
from deepspeed.runtime.pipe import schedule
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR

from oobleck.execution.dataloader import OobleckTrainDataLoader
from oobleck.module.layer import Layer
from oobleck.module.model import OobleckModel
from oobleck.module.layer import is_checkpointable
from oobleck.execution.utils import (
    zero_grads,
    run_once,
    DTYPE_TO_ID,
    ID_TO_DTYPE,
)
from oobleck.planning.pipeline_spec import PipelineSpec
from oobleck.utils.timer import OobleckTimer, measure_time

from transformers import TrainingArguments


class OobleckPipelineSchedule(schedule.TrainSchedule):
    """A schedule for training a batch using pipeline parallelism.

    Unlike existing :class:`deepspeed.runtime.pipe.schedule.TrainSchedule`,
    :class:`OobleckPipelineSchedule` decouples allreduce synchronization and optimizer step
    from pipeline execution and only schedules computation part and intermediate p2p operations.

    reducing (tied) gradients and optimizer step must be done separately.
    """

    def steps(self):
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(schedule.RecvActivation(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(schedule.SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(schedule.SendActivation(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(schedule.RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(schedule.LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(schedule.ForwardPass(curr_buffer))
                else:
                    cmds.append(schedule.BackwardPass(curr_buffer))

            # No reduce and optimizer step here at the end of the batch

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class PipelineExecutionMixin(object):
    def __init__(
        self,
        model_layers: List[Layer],
        training_args: TrainingArguments,
        dataloader: OobleckTrainDataLoader,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_layers = model_layers
        self.training_args = training_args
        self.dataloader = dataloader
        self.device = torch.device("cuda")

        self.reset_data_iterator()

        # store checkpointability for each layer
        for layer in self.model_layers:
            layer.set_checkpointable(
                not self.is_last_stage() and is_checkpointable(layer)
            )

        # stores the loss for the current microbatch being processed
        self.loss: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]] = None

        # stores the loss for the entire batch
        self.total_loss: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]] = None

        self.micro_steps = 0
        self.global_steps = 0
        self.global_samples = 0

        # TODO: use HF arguments to initialize properly
        parameters = list(
            itertools.chain(*[list(layer.parameters()) for layer in self.model_layers])
        )
        self.optimizer = FusedAdam(
            parameters,
            self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
            adam_w_mode=True,
        )
        num_training_steps = len(self.dataloader)
        self.lr_scheduler = WarmupLR(
            self.optimizer, self.training_args.get_warmup_steps(num_training_steps)
        )

    def reset_data_iterator(self):
        self.data_iterator = iter(self.dataloader)
        logger.info("iterator reset")

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2454
    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            data = data.clone().detach().to(self.device)
            data.requires_grad = data.is_floating_point()
            return data
        return data

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2472
    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        return tuple(self._prepare_input(t) for _, t in inputs.items())

    @instrument_w_nvtx
    @measure_time("execution_load_microbatch")
    def load_microbatch(self, buffer_id: int):
        assert (
            self.is_first_stage() or self.is_last_stage()
        ), "load_microatch can only be called at either the first stage or the last stage."

        if self.is_first_stage():
            batch = next(self.data_iterator)
            self.pipe_buffers["inputs"][buffer_id] = self._prepare_inputs(batch)

    @instrument_w_nvtx
    @measure_time("execution/forward")
    def forward_pass(self, buffer_id: int):
        inputs: tuple[torch.Tensor] = self.pipe_buffers["inputs"][buffer_id]
        zero_grads(inputs)

        # XXX Hack
        # Some tensor might be converted from torch.Size().
        # Convert it to torch.Size so that forward can be executed
        inputs: tuple[Union[torch.Size, torch.Tensor]] = tuple(
            [
                torch.Size(input.tolist())
                if input.dim() == 1
                and input.data[0] == self.training_args.per_device_train_batch_size
                else input
                for input in inputs
            ]
        )

        # Execute forward
        for layer in self.model_layers:
            inputs = layer(*inputs)

        outputs = inputs

        # Optionally compute loss on the last stage
        if self.is_last_stage():
            self.loss = outputs["loss"]

            if isinstance(self.loss, torch.Tensor):
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
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
                    else torch.LongTensor(data=output).to(self.device)
                    for output in outputs
                ]
            )

            self.pipe_buffers["outputs"][buffer_id] = outputs

    @instrument_w_nvtx
    @measure_time("execution/backward")
    def backward_pass(self, buffer_id: int):
        if self.is_last_stage():
            loss = self.loss
            loss.backward()
        else:
            output_tensors: Tuple[torch.Tensor] = self.pipe_buffers["outputs"][
                buffer_id
            ]
            output_tensors = tuple([t for t in output_tensors if t.requires_grad])
            grad_tensors: Tuple[torch.Tensor] = self.grad_recv_buf

            # Oobleck sharded model always returns tuple with tensors and torch.Size.
            assert len(output_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=output_tensors, grad_tensors=grad_tensors)

        # Free up memory from the output of forward()
        self.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None

    @instrument_w_nvtx
    @measure_time("execution/step")
    def optimizer_step(self, lr_kwargs=None):
        # amp enable check: gradient clipping
        self.optimizer.step()

        overflow = (
            self.optimizer.overflow if hasattr(self.optimizer, "overflow") else False
        )
        if not overflow:
            self.lr_scheduler.step(**(lr_kwargs or {}))


class PipelineCommunicationMixin(object):
    def __init__(self, process_group: ProcessGroup, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda")
        self.process_group = process_group
        self.my_rank = dist.get_rank(self.process_group)
        self.prev_rank: Optional[int] = None
        self.next_rank: Optional[int] = None

        self.num_pipe_buffers: int = 0
        self.pipe_buffers: Dict[str, Tuple[torch.Tensor]] = {
            "inputs": [],  # batch input and received activations
            "labels": [],  # labels from batch input
            "outputs": [],  # activations
        }

        # initialized in :func:`oobleck.execution.PipelineCommunicationMixin.recv_activations`.
        self.activation_recv_buf: Optional[Tuple[torch.Tensor]] = None
        # initialized in :func:`oobleck.execution.PipelineCommunicationMixin.recv_gradients`.
        self.grad_recv_buf: Optional[Tuple[torch.Tensor]] = None

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.
        This method only reserves slots and does not allocate tensors.
        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def _send(
        self, tensor: torch.Tensor, dest_rank: int, async_op: bool = False
    ) -> Work:
        dest_global_rank = dist.get_global_rank(self.process_group, dest_rank)
        return (
            dist.isend(tensor, dest_global_rank, self.process_group)
            if async_op
            else dist.send(tensor, dest_global_rank, self.process_group)
        )

    def _recv(
        self, tensor: torch.Tensor, src_rank: int, async_op: bool = False
    ) -> Work:
        src_global_rank = dist.get_global_rank(self.process_group, src_rank)
        return (
            dist.irecv(tensor, src_global_rank, self.process_group)
            if async_op
            else dist.recv(tensor, src_global_rank, self.process_group)
        )

    @run_once
    def _send_activation_meta(self, buffer: Tuple[torch.Tensor], receiver_rank: int):
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
        assert isinstance(buffer, tuple), f"Could not send meta type {type(buffer)}."
        count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
        self._send(count_tensor, receiver_rank)
        for tensor in buffer:
            assert isinstance(tensor, torch.Tensor)
            send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
            send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(
                self.device
            )
            send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
            send_req_grad = torch.LongTensor(
                data=[1 if tensor.requires_grad else 0]
            ).to(self.device)
            self._send(send_ndims, receiver_rank)
            self._send(send_dtype, receiver_rank)
            self._send(send_shape, receiver_rank)
            self._send(send_req_grad, receiver_rank)

    @measure_time("comm/send_activations")
    def send_activations(self, buffer_id: int):
        outputs: Tuple[torch.Tensor] = self.pipe_buffers["outputs"][buffer_id]
        self._send_activation_meta(outputs, self.next_rank)

        assert isinstance(outputs, tuple)
        for buffer in outputs:
            assert isinstance(buffer, torch.Tensor)
            self._send(buffer, self.next_rank)

    @measure_time("comm/recv_activations")
    def recv_activations(self, buffer_id: int):
        def create_receive_buffer(sender_rank: int) -> Tuple[torch.Tensor]:
            """Receive metadata about upcoming p2p transfers and return allocated buffer.

            Metadata is communicated in this order:
                * num_tensors in tensor tuple
                foreeach tensor in buffer:
                    * ndims
                    * dtype
                    * shape
                    * requires_grad
            """
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            self._recv(count_tensor, sender_rank)
            num_tensors = count_tensor.item()
            buffers: List[torch.Tensor] = []
            for _ in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                self._recv(recv_ndims, sender_rank)
                recv_ndims = recv_ndims.item()

                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                self._recv(recv_dtype, sender_rank)
                recv_dtype = ID_TO_DTYPE[recv_dtype.item()]

                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                self._recv(recv_shape, sender_rank)
                recv_shape = recv_shape.tolist()

                recv_req_grad = torch.LongTensor(data=[0]).to(self.device)
                self._recv(recv_req_grad, sender_rank)
                recv_req_grad = True if recv_req_grad.item() == 1 else False

                buffers.append(
                    torch.zeros(
                        recv_shape,
                        device=self.device,
                        dtype=recv_dtype,
                        requires_grad=recv_req_grad,
                    )
                )
            return tuple(buffers)

        if self.activation_recv_buf is None:
            self.activation_recv_buf = create_receive_buffer(self.prev_rank)

        assert isinstance(self.activation_recv_buf, tuple)
        recvd: List[Optional[torch.Tensor]] = [None] * len(self.activation_recv_buf)
        for idx, buffer in enumerate(self.activation_recv_buf):
            assert torch.is_tensor(buffer)
            self._recv(buffer, self.prev_rank)
            recvd[idx] = buffer.clone().detach()
            recvd[idx].requires_grad = buffer.requires_grad

        self.pipe_buffers["inputs"][buffer_id] = tuple(recvd)

    @measure_time("comm/send_gradients")
    def send_gradients(self, buffer_id: int):
        inputs = self.pipe_buffers["inputs"][buffer_id]
        assert isinstance(inputs, tuple)

        for buffer in inputs:
            # Skip tensors that will not produce a gradient
            if not buffer.requires_grad:
                assert buffer.grad is None
                continue
            assert buffer.grad is not None
            self._send(buffer.grad, self.prev_rank)

        # We can free up the input buffer now
        self.pipe_buffers["inputs"][buffer_id] = None

    @measure_time("comm/recv_gradients")
    def recv_gradients(self, buffer_id: int):
        def create_gradients_buffer(
            tensors: Tuple[torch.Tensor],
        ) -> Tuple[torch.Tensor]:
            assert isinstance(tensors, tuple)
            buffers: List[torch.Tensor] = []
            for tensor in tensors:
                assert isinstance(tensor, torch.Tensor)
                if tensor.requires_grad:
                    buffers.append(torch.zeros_like(tensor))

            return buffers

        outputs = self.pipe_buffers["outputs"][buffer_id]
        assert isinstance(outputs, tuple)

        # Allocate gradients if necessary
        if self.grad_recv_buf is None:
            self.grad_recv_buf = create_gradients_buffer(outputs)

        for buffer in self.grad_recv_buf:
            self._recv(buffer, self.next_rank)


# A map of PipeInstruction types to methods. Each method will be executed with the
# kwargs provided to the PipeInstruction from the scheduler.
INSTRUCTION_MAP = {
    schedule.OptimizerStep: PipelineExecutionMixin.optimizer_step,
    schedule.LoadMicroBatch: PipelineExecutionMixin.load_microbatch,
    schedule.ForwardPass: PipelineExecutionMixin.forward_pass,
    schedule.BackwardPass: PipelineExecutionMixin.backward_pass,
    schedule.SendActivation: PipelineCommunicationMixin.send_activations,
    schedule.RecvActivation: PipelineCommunicationMixin.recv_activations,
    schedule.SendGrad: PipelineCommunicationMixin.send_gradients,
    schedule.RecvGrad: PipelineCommunicationMixin.recv_gradients,
}


class OobleckPipeline(PipelineExecutionMixin, PipelineCommunicationMixin):
    """
    A realization of :class:`oobleck.planning.pipeline_spec.PipelineSpec`.
    It includes model to run, communication groups for pipeline execution,
    required functions for pipeline parallel execution of one pipeline.

    Note that it only communicates within the given process_group with local ranks,
    not global ranks.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        model: OobleckModel,
        dataloader: OobleckTrainDataLoader,
        process_group: ProcessGroup,
        training_args: TrainingArguments,
    ):
        self.layer_spec = spec.layer_spec
        self.model = model
        self.total_num_layers = len(model.model)
        self.timer = OobleckTimer()

        my_rank = dist.get_rank(process_group)
        model_layers = [
            layer.to("cuda")
            for layer, layer_rank in zip(model.model, self.layer_spec)
            if my_rank == layer_rank
        ]

        super().__init__(
            model_layers=model_layers,
            dataloader=dataloader,
            training_args=training_args,
            process_group=process_group,
        )

        logger.info(
            f"Creating pipeline ({len(spec.optimal_plan.stages)} stages): "
            f"{spec.optimal_plan.stages}"
        )

    def write_samples_logs(self):
        lr = next(
            iter(
                [
                    param_group["lr"]
                    for param_group in self.optimizer.param_groups
                    if "lr" in param_group
                ]
            ),
            0.0,
        )
        loss = self.total_loss.mean().item() if self.is_last_stage() else -1
        self.total_loss = None

        self.timer.write_events([(f"samples/lr", lr, self.global_steps)])
        self.timer.write_events([(f"samples/train_loss", loss, self.global_steps)])

    def train(self):
        assert (
            dist.get_rank(self.process_group) >= 0
        ), "This pipeline is not what I am involved in."
        unique_ranks = list(dict.fromkeys(self.layer_spec).keys())
        my_rank_index = unique_ranks.index(self.my_rank)
        self.train_schedule = OobleckPipelineSchedule(
            self.dataloader.num_my_microbatches,
            len(unique_ranks),  # total num stages
            my_rank_index,
        )
        self.prev_rank = my_rank_index - 1
        self.next_rank = my_rank_index + 1

        self._exec_schedule(self.train_schedule)

        self.global_steps += 1
        self.global_samples += (
            self.dataloader.num_total_microbatches
            * self.training_args.per_device_train_batch_size
        )

        self.write_samples_logs()

    def is_first_stage(self):
        return self.model_layers[0].index == 0

    def is_last_stage(self):
        return self.model_layers[-1].index == self.total_num_layers - 1

    def _exec_schedule(self, pipe_schedule: schedule.PipeSchedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                _exec_instr = MethodType(INSTRUCTION_MAP[type(cmd)], self)
                _exec_instr(**cmd.kwargs)
