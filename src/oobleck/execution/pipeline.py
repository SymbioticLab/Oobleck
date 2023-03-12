import torch
import itertools

from datetime import datetime
from types import MethodType
from typing import Union, Any, Dict, Mapping, Tuple, List, Optional

from torch.utils.checkpoint import checkpoint as checkpoint_fn

from torch.distributed import ProcessGroup, Work
from deepspeed import comm as dist
from deepspeed.utils import logger, instrument_w_nvtx, RepeatingLoader
from deepspeed.runtime.pipe import schedule
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.monitor.config import get_monitor_config
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR

from oobleck.execution.dataloader import OobleckDataLoader
from oobleck.module.model import OobleckModel
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
    @staticmethod
    def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
        if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
            return False
        if next(layer.parameters(), None) is None:
            return False
        return True

    def __init__(self):
        super().__init__()

        # store checkpointability for each layer
        for layer in self.model_layers:
            layer.checkpointable = (
                not self.is_last_stage()
                and PipelineExecutionMixin.is_checkpointable(layer)
            )

        # stores the loss for the current microbatch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        # stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

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
        num_training_steps = 2000  # TODO: fix this
        self.lr_scheduler = WarmupLR(
            self.optimizer, self.training_args.get_warmup_steps(num_training_steps)
        )

    def bfloat16_enabled(self):
        # TODO: modify this
        return False

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

        batch = next(self.data_iterator)
        labels = {"labels": batch.pop("labels")}

        if self.is_first_stage():
            self.pipe_buffers["inputs"][buffer_id] = self._prepare_inputs(batch)

        if self.is_last_stage():
            self.pipe_buffers["labels"][buffer_id] = self._prepare_inputs(labels)

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
            if self.is_last_stage() and layer == self.model_layers[-1]:
                # add labels in input if it is the last layer:
                inputs = inputs + self.pipe_buffers["labels"][buffer_id]

            if layer.checkpointable:
                inputs = checkpoint_fn(layer, *inputs)
            else:
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
            # TODO: gradient accumulation step scale loss
            loss.backward()
        else:
            output_tensors: Tuple[torch.Tensor] = self.pipe_buffers["outputs"][
                buffer_id
            ]
            output_tensors = tuple([t for t in output_tensors if t.requires_grad])
            grad_tensors: Tuple[torch.Tensor] = self.grad_recv_buf

            if self.bfloat16_enabled() and not self.is_last_stage():
                # manually call because we don't call optimizer.backward()
                self.optimizer.clear_lp_grads()

            # Oobleck sharded model always returns tuple with tensors and torch.Size.
            assert len(output_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=output_tensors, grad_tensors=grad_tensors)

            if self.bfloat16_enabled() and not self.is_last_stage():
                # manually call because we don't call optimizer.backward()
                self.optimizer.update_hp_grads(clear_lp_grads=False)

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

        self.global_steps += 1
        self.global_samples += 0  # TODO: fix this

        if self.monitor:
            # TODO :log lr
            # TODO: log train loss
            pass


class PipelineCommunicationMixin(object):
    def __init__(self):
        super().__init__()

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
        return (
            dist.isend(tensor, dest_rank, self.process_group)
            if async_op
            else dist.send(tensor, dest_rank, self.process_group)
        )

    def _recv(
        self, tensor: torch.Tensor, src_rank: int, async_op: bool = False
    ) -> Work:
        return (
            dist.irecv(tensor, src_rank, self.process_group)
            if async_op
            else dist.recv(tensor, src_rank, self.process_group)
        )

    @instrument_w_nvtx
    @measure_time("comm/reduce_gradients")
    def reduce_gradients(self):
        pass

    @instrument_w_nvtx
    @measure_time("cpmm/reduced_tied_gradients")
    def reduce_tied_gradients(self):
        pass

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
    schedule.ReduceGrads: PipelineCommunicationMixin.reduce_gradients,
    schedule.ReduceTiedGrads: PipelineCommunicationMixin.reduce_tied_gradients,
    schedule.LoadMicroBatch: PipelineExecutionMixin.load_microbatch,
    schedule.ForwardPass: PipelineExecutionMixin.forward_pass,
    schedule.BackwardPass: PipelineExecutionMixin.backward_pass,
    schedule.SendActivation: PipelineCommunicationMixin.send_activations,
    schedule.RecvActivation: PipelineCommunicationMixin.recv_activations,
    schedule.SendGrad: PipelineCommunicationMixin.send_gradients,
    schedule.RecvGrad: PipelineCommunicationMixin.recv_gradients,
}


class Pipeline(PipelineExecutionMixin, PipelineCommunicationMixin):
    """
    A realization of :class:`oobleck.planning.pipeline_spec.PipelineSpec`.
    It includes model to run, communication groups for pipeline execution,
    required functions for pipeline parallel execution of one pipeline.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        model: OobleckModel,
        dataloader: OobleckDataLoader,
        process_group: ProcessGroup,
        training_args: TrainingArguments,
    ):
        assert spec.num_nodes == dist.get_world_size(process_group), (
            f"PipelineSpec (# nodes: {spec.num_nodes}) does not match with "
            f"the given ProcessGroup size ({dist.get_world_size(process_group)})"
        )

        self.process_group = process_group
        self.local_rank = dist.get_rank(self.process_group)
        self.device = torch.device("cuda")

        self.model = model
        self.plan = spec.create_optimal_plan(model)

        # Let it raise an exception if there is no assigned stage by not adding default value to next().
        stage_spec = next(filter(lambda s: self.local_rank in s.ranks, self.plan))
        self.layer_start_index, self.layer_end_index = stage_spec.get_layer_indices()
        self.model_layers = self.model.model[
            self.layer_start_index : self.layer_end_index
        ]
        self.total_num_layers = len(self.model.model)
        self.total_num_stages = len(self.plan)

        my_stage_index = self.plan.index(stage_spec)
        my_rank_index = stage_spec.ranks.index(self.local_rank)
        self.prev_rank = (
            self.plan[my_stage_index - 1].ranks[my_rank_index]
            if not self.is_first_stage()
            else -1
        )
        self.next_rank = (
            self.plan[my_stage_index + 1].ranks[my_rank_index]
            if not self.is_last_stage()
            else -1
        )

        self.training_args = training_args
        super().__init__()

        self.model = model
        self.dataloader = dataloader
        self.data_iterator = iter(self.dataloader)

        if dist.get_rank() == 0:
            self.monitor = MonitorMaster(
                get_monitor_config(
                    {
                        "tensorboard": {
                            "enabled": True,
                            "output_path": "/tmp/oobleck/tensorboard/",
                            "job_name": f"{self.model.model_name}-{datetime.now().strftime('%m-%d-%Y,%H:%M:%S')}",
                        }
                    }
                )
            )
            self.timers = OobleckTimer(self.monitor)
        else:
            self.monitor = None
            self.timers = None

    def train(self):
        train_schedule = OobleckPipelineSchedule(
            self.training_args.per_device_train_batch_size,
            self.total_num_stages,
            self.local_rank,
        )
        self._exec_schedule(train_schedule)

        # TODO: fix this
        self.agg_train_loss = 0

        if self.timers:
            self.timers.log(
                [
                    "execution/load_microbatch",
                    "execution/forward",
                    "execution/backward",
                    "execution/step",
                    "comm/reduce_gradients",
                    "comm/reduced_tied_gradients",
                    "comm/send_activations",
                    "comm/recv_activations",
                    "comm/send_gradients",
                    "comm/recv_gradients",
                    "samples/lr",
                    "samples/train_loss",
                ],
                self.global_steps,
            )

        return self.agg_train_loss

    def is_first_stage(self):
        return self.layer_start_index == 0

    def is_last_stage(self):
        return self.layer_end_index == self.total_num_layers

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


if __name__ == "__main__":
    from oobleck.execution.dataset import OobleckDataset
    from oobleck.module.model import OobleckModel

    dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
    model = OobleckModel("gpt2", dataset.trace_input_names)

    from oobleck.execution.dataloader import OobleckDataLoader
    from transformers import TrainingArguments

    args = TrainingArguments(output_dir="/tmp/output")
    train_dataloader = RepeatingLoader(
        OobleckDataLoader(
            dataset.dataset["train"],
            args.per_device_train_batch_size,
            dataset.data_collator,
            args,
        )
    )

    from oobleck.planning.pipeline_spec import PipelineSpec

    pipe_spec = PipelineSpec(2, 1)

    import os
    from deepspeed import comm as dist

    os.environ["RANK"] = "1"
    os.environ["LOCAL_RANK"] = "1"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25400"

    # if dist is already initialized, destroy it.
    if dist.is_initialized():
        dist.destroy_process_group()

    dist.init_distributed("nccl")
    pg = dist.new_group([0, 1])

    from oobleck.execution.pipeline import Pipeline

    pipeline = Pipeline(pipe_spec, model, train_dataloader, pg, args)
    for i in range(30):
        pipeline.train()
