import torch

from datetime import datetime
from types import MethodType

from torch.distributed import ProcessGroup
from deepspeed import comm as dist
from deepspeed.utils import logger, instrument_w_nvtx
from deepspeed.runtime.pipe import schedule, p2p
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.monitor.config import get_monitor_config

from oobleck.execution.dataloader import OobleckDataLoader
from oobleck.execution.model import OobleckModel
from oobleck.execution.utils import zero_grads
from oobleck.planning.pipeline_spec import PipelineSpec
from oobleck.utils.timer import OobleckTimer, measure_time

from transformers import TrainingArguments


class PipelineExecutionMixin(object):
    def __init__(self):
        super().__init__()

        # stores the loss for the current microbatch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        self.forward_outputs = []

        # stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        self.loss_model = None if self.is_last_stage() else torch.nn.CrossEntropyLoss()
        self.optimizer = None

    def bfloa16_enabled(self):
        # TODO: modify this
        return False

    @instrument_w_nvtx
    @measure_time
    def load_microbatch(self, buffer_id: int):
        logger.info(__name__)
        batch = next(self.data_iterator)

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], tuple)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers["inputs"][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers["labels"][buffer_id] = loaded

    @instrument_w_nvtx
    @measure_time
    def forward_pass(self, buffer_id: int):
        logger.info(__name__)
        # Prepare inputs
        if isinstance(self.pipe_buffers["inputs"][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers["inputs"][buffer_id])
        else:
            inputs = self.pipe_buffers["inputs"][buffer_id].clone()
        zero_grads(inputs)

        # Execute forward
        for layer in self.model_layers:
            inputs = layer(*inputs)

        outputs = inputs
        self.pipe_buffers["outputs"][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            assert self.loss_model
            # "labels" buffer is filled in `.load_microbatch()` function
            labels = self.pipe_buffers["labels"][buffer_id]
            self.loss = self.loss_model(outputs, labels)

        if isinstance(self.loss, torch.Tensor):
            self.fwd_outputs.append(self.loss.detach())

            if self.total_loss is None:
                self.total_loss = torch.zeros_like(self.loss)
            self.total_loss += self.loss.detach()
        else:
            self.fwd_outputs.append([l.detach() for l in self.loss])

            if self.total_loss is None:
                self.total_loss = [torch.zeros_like(l) for l in self.loss]
            for idx, l in enumerate(self.loss):
                self.total_loss[idx] += l.detach()

    @instrument_w_nvtx
    @measure_time
    def backward_pass(self, buffer_id: int):
        logger.info(__name__)
        loss = self.loss

        if self.is_last_stage():
            # TODO: gradient accumulation step scale loss
            loss.backward()
        else:
            outputs = self.pipe_buffers["outputs"][buffer_id]
            grad_tensors = self.grad_layer

            if self.bfloa16_enabled() and not self.is_last_stage():
                # manually call because we don't call optimizer.backward()
                self.optimizer.clear_lp_grads()

            # This handles either a single tensor or tuple of tensors.
            if isinstance(outputs, tuple):
                out_tensors = [t for t in outputs if t.is_floating_point()]
                assert len(out_tensors) == len(grad_tensors)
                torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
            else:
                torch.autograd.backward(
                    tensors=(outputs,), grad_tensors=(grad_tensors,)
                )

            if self.bfloat16_enabled() and not self.is_last_stage():
                # manually call because we don't call optimizer.backward()
                self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up memory from the output of forward()
        self.pipe_buffers["output_tensors"][buffer_id] = None
        self.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None

    @instrument_w_nvtx
    @measure_time
    def optimizer_step(self, lr_kwargs=None):
        logger.info(__name__)
        # amp enable check: gradient clipping
        self.optimizer_step()

        overflow = (
            self.optimizer.overflow if hasattr(self.optimizer, "overflow") else False
        )
        if not overflow:
            self.lr_scheduler.step(**(lr_kwargs or {}))


class PipelineCommunicationMixin(object):
    def __init__(self):
        super().__init__()

        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            "inputs": [],  # batch input and received activations
            "labels": [],  # labels from batch input
            "outputs": [],  # activations
            "output_tensors": [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.first_output_send = True
        self.first_gradient_send = True

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

    @instrument_w_nvtx
    @measure_time
    def reduce_gradients(self):
        logger.info(__name__)
        pass

    @instrument_w_nvtx
    @measure_time
    def reduce_tied_gradients(self):
        logger.info(__name__)
        pass

    def send_activation(self, buffer_id: int):
        logger.info(__name__)
        pass

    def recv_activation(self, buffer_id: int):
        logger.info(__name__)
        pass

    def send_gradients(self, buffer_id: int):
        logger.info(__name__)
        pass

    def recv_gradients(self, buffer_id: int):
        logger.info(__name__)
        pass


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
            f"PipelineSpec (# nodes: {len(spec.num_nodes)}) does not match with "
            f"the given ProcessGroup size ({dist.get_world_size(process_group)})"
        )

        self.process_group = process_group
        self.local_rank = dist.get_rank(self.process_group)
        self.device = torch.device("cuda")

        self.model = model
        self.plan = spec.create_optimal_plan(model)
        # Let it raise an exception if there is no assigned stage by not adding default value.
        stage_spec = next(filter(lambda s: self.local_rank in s.ranks, self.plan))
        self.layer_start_index, self.layer_end_index = stage_spec.get_layer_indices()
        self.model_layers = [self.model.model.split_gm.children()][
            self.layer_start_index : self.layer_end_index
        ]
        self.total_num_layers = len([self.model.model.split_gm.children()])
        self.total_num_stages = len(self.plan)

        super().__init__()

        self.model = model
        self.dataloader = dataloader

        self.training_args = training_args
        if dist.get_rank() == 0:
            self.monitor = MonitorMaster(
                get_monitor_config(
                    {
                        "tensorboard": {
                            "enabled": True,
                            "output_path": "/tmp/oobleck/tensorboard/",
                            "job_name": f"{self.model_name}-{datetime.now().strftime('%m-%d-%Y,%H:%M:%S')}",
                        }
                    }
                )
            )
            self.timers = OobleckTimer(self.monitor)
        else:
            self.timers = None

    def train(self):
        train_schedule = schedule.TrainSchedule(
            self.training_args.per_device_train_batch_size,
            self.total_num_stages,
            self.local_rank,
        )
        self._exec_schedule(train_schedule)

    def is_first_stage(self):
        return self.layer_start_index == 0

    def is_last_stage(self):
        return self.layer_end_index == self.total_num_layers - 1

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: PipelineExecutionMixin.optimizer_step,
        schedule.ReduceGrads: PipelineCommunicationMixin.reduce_gradients,
        schedule.ReduceTiedGrads: PipelineCommunicationMixin.reduce_tied_gradients,
        schedule.LoadMicroBatch: PipelineExecutionMixin.load_microbatch,
        schedule.ForwardPass: PipelineExecutionMixin.forward_pass,
        schedule.BackwardPass: PipelineExecutionMixin.backward_pass,
        schedule.SendActivation: PipelineCommunicationMixin.send_activation,
        schedule.RecvActivation: PipelineCommunicationMixin.recv_activation,
        schedule.SendGrad: PipelineCommunicationMixin.send_gradients,
        schedule.RecvGrad: PipelineCommunicationMixin.recv_gradients,
    }

    def _exec_schedule(self, pipe_schedule: schedule.PipeSchedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                _exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                _exec_instr(self, **cmd.kwargs)


if __name__ == "__main__":
    from oobleck.execution.dataset import OobleckDataset
    from oobleck.execution.model import OobleckModel

    dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
    model = OobleckModel("gpt2", dataset.trace_input_names)

    from oobleck.execution.dataloader import OobleckDataLoader
    from transformers import TrainingArguments

    args = TrainingArguments(output_dir="/tmp/output")
    dataloader = OobleckDataLoader(dataset, dataset.data_collator, args)

    from oobleck.planning.pipeline_spec import PipelineSpec

    pipe_spec = PipelineSpec(2, 1)

    import os
    from deepspeed import comm as dist

    os.environ["RANK"] = "1"
    os.environ["LOCAL_RANK"] = "1"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6900"

    # if dist is already initialized, destroy it.
    if dist.is_initialized():
        dist.destroy_process_group()

    dist.init_distributed("nccl")
    pg = dist.new_group([0, 1])

    from oobleck.execution.pipeline import Pipeline

    pipeline = Pipeline(pipe_spec, model, dataloader, pg, args)
    pipeline.train()
