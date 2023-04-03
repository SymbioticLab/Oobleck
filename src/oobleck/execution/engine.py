import os
import pyomo.environ as pyomo
import torch.distributed
import gc
import sys
import time

from ast import literal_eval
from typing import Optional, Dict, Tuple, List, Any, TypeVar
from torch.distributed import ProcessGroup

from deepspeed import comm as dist
from deepspeed.utils import logger

from oobleck.elastic.client import RedisClient
from oobleck.execution.dataset import OobleckDataset
from oobleck.module.layer import Layer
from oobleck.module.model import OobleckModel
from oobleck.planning.pipeline_spec import PipelineSpec
from oobleck.planning.instantiator import (
    HeterogeneousPipelineExecutionPlan,
    PipelineInstantiator,
)

from oobleck.execution.dataloader import OobleckTrainDataLoader
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.utils.timer import OobleckTimer, measure_time

from transformers import TrainingArguments

T = TypeVar("T", bound="OobleckEngine")


class DynamicReconfigurationMixin(object):
    """
    Oobleck dynamic reconfiguration Mixin.
    """

    def __init__(self):
        super().__init__()

    def init_reconfiguration(self: T):
        logger.info("Reconfiguration start...")

        start = time.time()
        self.init_distributed()
        # We now have less number of GPUs.

        logger.info("new world: %s", self.world_info)

        execution_plan = self.create_execution_plan(
            self.pipeline_specs, self.world_size, self.global_num_microbatch, True
        )

        old_local_rank = self.my_pipeline.my_rank
        old_spec: List[int] = self.my_pipeline.layer_spec

        # 1. First, advertise that I have the layers
        self.redis.append_my_rank_to_layers(
            self.rank, [l.index for l in self.my_pipeline.model_layers]
        )
        dist.barrier()

        # get who owns which layers
        having_layers = self.redis.get_all_having_layers()
        logger.info("having_layers: %s", having_layers)
        # check whether there is a layer owned by nobody
        assert len(having_layers) == len(self.model.model) and all(
            l for l in having_layers.values()
        ), "Some layers are not owned by any node."

        # =======================================================
        # =================== Reconfiguration ===================
        # =======================================================
        # 2. find me from execution plan
        def get_my_pipeline_spec() -> Tuple[PipelineSpec, int]:
            total_num_nodes_used = 0
            for spec in execution_plan.pipeline_specs:
                for _ in range(execution_plan.num_instances_set[spec]):
                    if self.rank in range(
                        total_num_nodes_used, total_num_nodes_used + spec.num_nodes
                    ):
                        return spec, self.rank - total_num_nodes_used
                    total_num_nodes_used += spec.num_nodes
            raise ValueError("Cannot find my pipeline spec")

        # local_rank: my location in the pipeline process group starting from 0.
        # new_pipeline_spec.layer_spec has rank assigned to each layer,
        # which includes my local rank.
        new_pipeline_spec, new_local_rank = get_my_pipeline_spec()

        # 3. find missing layers
        # Missing layers mean layers that I didn't own but assigned to me in a new spec.
        # If any, report it to Redis.
        assert (
            len(self.model.model) == len(new_pipeline_spec.layer_spec) == len(old_spec)
        ), "Number of layers in the model is inconsistent with the number of layers in the pipeline spec."
        missing_layers: List[Layer] = [
            layer
            for layer, old_layer_rank, new_layer_rank in zip(
                self.model.model, old_spec, new_pipeline_spec.layer_spec
            )
            if new_layer_rank == new_local_rank and old_layer_rank != old_local_rank
        ]
        self.redis.append_missing_layers(self.rank, [l.index for l in missing_layers])
        dist.barrier()

        if missing_layers:
            logger.info(
                f"Getting {len(missing_layers)} missing layers: "
                f"{[l.index for l in missing_layers]}"
            )
        else:
            logger.info(
                "No missing layer. Waiting for other rank reconfiguration to be done."
            )

        all_missing_layers: Dict[int, List[int]] = self.redis.get_all_missing_layers()
        logger.info("all missing layers: %s", all_missing_layers)
        # all_missing_layers: the key is the index of layers, and the value is the list of ranks that need the layer.
        # For example, {0: [0, 1, 2], 1: [0, 1, 2]} means that layer 0 and 1 are missing in rank 0, 1, and 2.
        # Iterate all layers, broadcast the layer.
        # Source of broadcast is the smallest rank in range(0, new_world_size) that is not in the list of ranks that need the layer.
        # If my rank (self.rank) is in the list of ranks that need the layer, then I will receive the layer.
        for layer_index, ranks in all_missing_layers.items():
            # Find the smallest rank in range(0, new_world_size) that is not in the list of ranks that need the layer.
            source_rank = min(set(having_layers[layer_index]), default=self.rank)
            if self.rank in ranks:
                # I need the layer.
                logger.info(f"Receiving layer {layer_index} from rank {source_rank}...")
            else:
                # I don't need the layer.
                # if I (self.rank) am the source rank, then I will send the layer.
                if self.rank == source_rank:
                    logger.info(f"Sending layer {layer_index} to ranks {ranks}...")
            target_layer = self.model.model[layer_index]

            # TODO: send optimizer state, too.
            # TODO: update opptimizer state dict. Modify class structure if needed.
            torch.distributed.broadcast_object_list(
                [list(target_layer.parameters())], src=source_rank
            )

        # Reinstantiate pipeline
        self.pipeline = self.instantiate_pipelines(execution_plan)

        end = time.time()
        logger.info("Reconfiguration time: %s s", end - start)


class FSDPMixin(object):
    """
    Oobleck Fully-Sharded Data Parallel (FSDP) Mixin
    for sharding and distributing a stage into multiple intra-node GPUs.

    TODO: implement it.
    """

    def __init__(self):
        super().__init__()


class DataSynchronizationMixin(object):
    """
    Oobleck model parameter synchronization across pipelines Mixin.
    :class:`oobleck.execution.pipeline.OobleckPipeline`
    """

    def __init__(self):
        super().__init__()

    def initialize_dp_process_groups(
        self, pipeline: OobleckPipeline, dp_layer_groups: List[ProcessGroup]
    ):
        assert len(dp_layer_groups) == len(
            pipeline.model_layers
        ), "Number of model layer is inconsistent with number of process groups."
        self.my_pipeline = pipeline
        self.dp_layer_groups = dp_layer_groups

    @measure_time("comm/reduce_gradients")
    def do_allreduce(self):
        for index, layer in reversed(list(enumerate(self.my_pipeline.model_layers))):
            layer.reduce_gradients(self.dp_layer_groups[index])


class OobleckEngine(
    DataSynchronizationMixin,
    DynamicReconfigurationMixin,
    FSDPMixin,
):
    """
    Oobleck distributed training execution engine based on DeepSpeed.
    It initializes several pipelines as needed and launch them in parallel across nodes.
    Heterogeneous pipeline might have different pipeline schedules, thus
    :class:`oobleck.execution.pipeline.OobleckPipeline` is responsible for pipeline task scheduling.

    Engine initialization has two parts: traditional `__init__` does distributed-agnostic
    initialization, while `init_distributed()` initializes distributed related arguments.
    `init_distributed()` is called in :class:`ElasicMonitorMixin` when it receives
    training begins.
    """

    def __init__(
        self,
        fault_tolerance_spec: int,
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
    ):
        assert (
            not dist.is_initialized()
        ), "torch.distributed must not be initialized when initializing OobleckEngine."

        super().__init__()

        # ==================================================================
        # Initialization agnostic to distributed
        # ==================================================================

        self.node_name: Tuple[str, int] = literal_eval(os.environ["NODE_NAME"])
        self.max_num_nodes = int(os.environ["MAX_NUM_NODES"])
        self.num_gpus_per_node = int(os.environ["NUM_GPUS_PER_NODE"])
        # Remove LOCAL_RANK env so that TrainingArgument does not
        # automatically initialize torch.distributed.
        self.local_rank = int(os.environ.pop("LOCAL_RANK", 0))
        training_args = TrainingArguments("/tmp/output", **training_args)
        if dist.is_initialized():
            dist.destroy_process_group()

        self.ft_spec = fault_tolerance_spec
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.training_args = training_args

        self.dataset = OobleckDataset(model_name, dataset_path, dataset_name)

        model_args = {} if model_args is None else model_args
        model_args["use_cache"] = False
        model_args["remove_unused_columns"] = False
        self.model = OobleckModel(
            model_name, self.dataset.sample, training_args, model_args
        )

        # create a list of pipelinespecs that can cover all nodes.
        # this is invariant and never changes over reconfiguration.
        self.pipeline_specs: List[PipelineSpec] = PipelineSpec.create(
            self.ft_spec, self.max_num_nodes, self.num_gpus_per_node, self.model
        )

        self.global_num_microbatch = training_args.gradient_accumulation_steps

    def init_distributed(self):
        if not hasattr(self, "redis") or self.redis is None:
            self.redis = RedisClient()
            self.redis.subscribe_reconfiguration()
        self.redis.reconfiguration_required = False

        self.world_info: Dict[Tuple[str, int], List[int]] = self.redis.get_world_info()
        self.world_size = sum(len(gpus) for gpus in self.world_info.values())

        if self.node_name not in self.world_info:
            logger.info(f"{self.node_name} is preempted. Out.")
            sys.exit(0)

        self.rank = self.world_info[self.node_name][self.local_rank]
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        # initiate distributed
        if dist.is_initialized():
            dist.destroy_process_group()
            del dist.cdb
            dist.cdb = None
            del self.tcpstore
            del self.pipeline
            gc.collect()

        if self.rank == 0:
            addr = self.node_name[0]
            self.tcpstore = torch.distributed.TCPStore(
                addr,
                self.master_port,
                world_size=self.world_size,
                is_master=True,
                timeout=torch.distributed.default_pg_timeout,
            )
        else:
            # Find the address of rank 0.
            rank0_addr = next(k for k, v in self.world_info.items() if 0 in v)[0]

            self.tcpstore = torch.distributed.TCPStore(
                rank0_addr,
                self.master_port,
                world_size=self.world_size,
                is_master=False,
                timeout=torch.distributed.default_pg_timeout,
            )

        self.master_port += 1

        # TODO: use MPI so that we don't have to use TCPStore.
        torch.distributed.init_process_group(
            "nccl", store=self.tcpstore, rank=self.rank, world_size=self.world_size
        )
        dist.init_distributed("nccl", dist_init_required=False)

        self.timer: OobleckTimer = OobleckTimer()

    # ==========================================================================================
    # Paper section 4.1. is implemented in oobleck.planning.pipeline_spec.PipelineSpec.
    # Paper section 4.2. is implemented in oobleck.planning.pipeline_spec.PipelineSpec.
    # Paper section 4.3. is implemented in oobleck.planning.instantiator.PipelineInstantiator.
    # ==========================================================================================

    def create_execution_plan(
        self,
        pipeline_specs: List[PipelineSpec],
        num_nodes: int,
        global_num_microbatch: int,
        throughput_oriented: bool = True,
    ) -> HeterogeneousPipelineExecutionPlan:
        instantiator = PipelineInstantiator(
            pipeline_specs, num_nodes, global_num_microbatch
        )

        return instantiator.get_best_execution_plan(throughput_oriented)

    def instantiate_pipelines(
        self,
        execution_plan: HeterogeneousPipelineExecutionPlan,
    ) -> OobleckPipeline:
        """Oobleck paper section 4.3. Instantiating Pipeline Templates implementation
        Instantiate given `PipelineSpec`s and create `OobleckPipeline`s.

        Args:
            pipeline_specs (List[PipelineSpec]): List of `PipelineSpec`s to be
                used for instantiation.
            num_nodes: int: Number of nodes.
            throughput_oriented (bool, optional): Whether throughput oriented or
                reconfiguration overhead oriented.
        """

        self.training_args.gradient_accumulation_steps = (
            execution_plan.get_my_number_of_microbatches(dist.get_rank())
        )
        self.epoch, self.step, consumed_samples = self.redis.get_training_progress()
        if consumed_samples != 0:
            logger.info(
                "Continuing training from (epoch, step, consumed_samples): "
                f"{self.epoch, self.step, consumed_samples}"
            )
        train_dataloader = OobleckTrainDataLoader(
            self.dataset.dataset["train"],
            self.training_args,
            self.global_num_microbatch,
            consumed_samples,
            self.epoch,
            self.dataset.data_collator,
        )

        pipeline, pipeline_ranks_list = execution_plan.instantiate(
            self.model, train_dataloader, self.training_args, self.step
        )

        # Reconstruct per-layer rank group for data parallelism from execution plan
        layer_dp_groups: List[ProcessGroup] = []
        for layer_index in range(len(pipeline.model.model)):
            ranks = [ranks[layer_index] for ranks in pipeline_ranks_list]
            dp_pg = dist.new_group(ranks)
            if self.rank in ranks:
                layer_dp_groups.append(dp_pg)
        assert len(layer_dp_groups) == len(pipeline.model_layers)

        self.initialize_dp_process_groups(pipeline, layer_dp_groups)

        return pipeline

    @measure_time("samples/iteration")
    def train_step(self, reset_iterator: bool):
        if reset_iterator:
            try:
                self.my_pipeline.train()
            except StopIteration:
                self.my_pipeline.reset_data_iterator()
                self.my_pipeline.train()
        else:
            self.my_pipeline.train()
        self.do_allreduce()
        self.my_pipeline.optimizer_step()

    def train(self):
        """
        Train my pipeline and synchronize gradients after each schedule is done
        until specified steps or epoch is reached.
        """

        def log():
            self.timer.log_throughput(
                self.global_num_microbatch
                * self.training_args.per_device_train_batch_size,
                self.world_size,
                "samples/iteration",
                self.my_pipeline.global_steps,
            )
            self.timer.log(
                [
                    "execution/forward",
                    "execution/backward",
                    "execution/step",
                    "comm/send_activations",
                    "comm/recv_activations",
                    "comm/send_gradients",
                    "comm/recv_gradients",
                    "comm/reduce_gradients",
                    "samples/lr",
                    "samples/train_loss",
                ],
                self.my_pipeline.global_steps,
            )

        self.master_port = 25400
        self.init_distributed()
        execution_plan = self.create_execution_plan(
            self.pipeline_specs, self.world_size, self.global_num_microbatch, True
        )
        self.pipeline = self.instantiate_pipelines(execution_plan)

        if self.training_args.max_steps > 0:
            for _ in range(self.step, self.training_args.max_steps):
                logger.info(f"[{self.my_pipeline.global_steps}] step")
                self.train_step(True)
                self.redis.set_training_progress(
                    0,
                    self.my_pipeline.global_steps,
                    self.my_pipeline.dataloader.batch_sampler.consumed_samples,
                )
                log()
                if self.redis.reconfiguration_required:
                    self.init_reconfiguration()
        else:
            for e in range(self.epoch, int(self.training_args.num_train_epochs)):
                num_steps = len(self.my_pipeline.dataloader)
                for _ in range(self.step, num_steps):
                    logger.info(f"[{self.my_pipeline.global_steps}] step")
                    self.train_step(False)
                    self.redis.set_training_progress(
                        e,
                        self.my_pipeline.global_steps,
                        self.my_pipeline.dataloader.batch_sampler.consumed_samples,
                    )
                    log()
                    if self.redis.reconfiguration_required:
                        self.init_reconfiguration()
                self.step = 0
                self.my_pipeline.reset_data_iterator()
