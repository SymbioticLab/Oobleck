import os
import math
import torch

from typing import Optional, Dict, List, Type, Any

from torch.distributed import ProcessGroup

from deepspeed import comm as dist
from deepspeed.utils import RepeatingLoader, logger

from oobleck.execution.dataset import OobleckDataset
from oobleck.module.model import OobleckModel
from oobleck.planning.pipeline_spec import (
    PipelineSpec,
    get_pipeline_specs,
)
from oobleck.elastic.client import ElasticWorkerClientMixin, ElasticClientMonitorMixin
from oobleck.execution.dataloader import OobleckDataLoader
from oobleck.execution.pipeline import OobleckPipeline

from transformers import TrainingArguments


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

    def do_allreduce(self):
        for index, layer in reversed(list(enumerate(self.my_pipeline.model_layers))):
            layer.reduce_gradients(self.dp_groups[index])


class OobleckEngine(
    ElasticClientMonitorMixin,
    ElasticWorkerClientMixin,
    DataSynchronizationMixin,
    FSDPMixin,
):
    """
    Oobleck distributed training execution engine based on DeepSpeed.
    It initializes several pipelines as needed and launch them in parallel across nodes.
    Heterogeneous pipeline might have different pipeline schedules, thus
    :class:`oobleck.execution.pipeline.Pipeline` is responsible for pipeline task scheduling.

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
    ):
        assert (
            not dist.is_initialized()
        ), "torch.distributed must not be initialized when initializing OobleckEngine."

        super().__init__()

        # ==================================================================
        # Initialization agnostic to distributed
        # ==================================================================

        self.node_name = os.environ["NODE_NAME"]
        self.max_num_nodes = int(os.environ["MAX_NUM_NODES"])
        self.num_gpus_per_node = int(os.environ["NUM_GPUS_PER_NODE"])
        # Remove LOCAL_RANK env so that TrainingArgument does not
        # automatically initialize torch.distributed.
        self.local_rank = int(os.environ.pop("LOCAL_RANK", 0))
        training_args = TrainingArguments("/tmp/output")
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
            model_name, self.dataset.trace_input_names, training_args, model_args
        )

        self.train_dataloader = RepeatingLoader(
            OobleckDataLoader(
                self.dataset.dataset["train"],
                self.training_args.per_device_train_batch_size,
                self.dataset.data_collator,
                self.training_args,
            )
        )

        # calculate minimum required # of GPUs to hold the parameters.
        # assume we use AMP Adam (having 12x memory bloat).
        # assume we don't consider activations, since we use activation recomputation.
        required_memory = self.model.total_num_params * 12
        required_min_gpus = math.ceil(
            required_memory / torch.cuda.get_device_properties("cuda:0").total_memory
        )
        self.required_min_nodes = math.ceil(required_min_gpus / self.num_gpus_per_node)

        # create a list of pipelinespecs that can cover all nodes.
        # this is invariant and never changes over reconfiguration.
        self.pipeline_specs: List[PipelineSpec] = get_pipeline_specs(
            self.ft_spec,
            self.required_min_nodes,
            self.max_num_nodes,
            self.num_gpus_per_node,
            self.model,
        )

    def init_distributed(self):
        # TODO: get world_info from etcd
        # world_info = self.get_world_info()
        # self.world_info = {"localhost1": [0], "localhost2": [1]}
        self.world_info = {"localhost1": [0], "localhost2": [1]}
        assert all(
            len(gpus) > 0 for gpus in self.world_info.values()
        ), "Some node has no GPUs."
        self.rank = self.world_info[self.node_name][self.local_rank]

        # use any node to get # of GPUs per node. They should all have the same #.
        self.num_gpus_per_node = len(next(iter(self.world_info.values())))
        assert all(
            len(gpus) == self.num_gpus_per_node for gpus in self.world_info.values()
        ), "Some node has different number of GPUs."

        # initiate distributed
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["WORLD_SIZE"] = str(
            sum(len(gpus) for gpus in self.world_info.values())
        )

        logger.info

        # TODO: get master information from etcd
        # master_info = self.get_torch_master_info()
        # os.environ["MASTER_IP"] = master_info[0]
        # os.environ["MASTER_PORT"] = str(master_info[1])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "25400"

        dist.init_distributed("nccl", auto_mpi_discovery=False)

        self.my_pipeline = self._deploy_pipelines()
        # self.fsdp_pipelines: List[OobleckPipeline] = list(
        #     p
        #     for p in self.pipelines
        #     if any(r in p.ranks for r in self.world_info[self.node_name])
        # )
        assert self.my_pipeline, "Cannot find a pipeline that I am belong to."
        # assert self.fsdp_pipelines, "Cannot find FSDP pipelines that I am belong to."

    def _deploy_pipelines(self) -> OobleckPipeline:
        """Calculate required number of heterogeneous pipelines that
        a linear combination of the pipelines fills the distributed world.
        """
        num_nodes = dist.get_world_size() // self.num_gpus_per_node
        assert num_nodes <= self.max_num_nodes, "World size is larger than max # nodes."

        # Current Alpha-level implementation: always prefer smaller pipelines.
        # TODO: analyze the best optimal combination that has the highest throughput.
        num_pipelines = [0] * len(self.pipeline_specs)
        num_pipelines[0] = math.floor(num_nodes / self.pipeline_specs[0].num_nodes)
        total_assigned_nodes = num_pipelines[0] * self.pipeline_specs[0].num_nodes
        assert (
            total_assigned_nodes <= num_nodes
        ), f"total assigned nodes {total_assigned_nodes} is not less than total given nodes {num_nodes}"

        smallest_non_zero_pipeline_index = 0
        while total_assigned_nodes < num_nodes:
            while (
                smallest_non_zero_pipeline_index < len(self.pipeline_specs)
                and num_pipelines[smallest_non_zero_pipeline_index] == 0
            ):
                smallest_non_zero_pipeline_index += 1

            if (
                smallest_non_zero_pipeline_index + 1 < len(self.pipeline_specs)
                and num_pipelines[smallest_non_zero_pipeline_index] > 0
            ):
                num_pipelines[smallest_non_zero_pipeline_index] -= 1
                num_pipelines[smallest_non_zero_pipeline_index + 1] += 1
                total_assigned_nodes += 1

        assert (
            sum(
                num_pipelines[i] * self.pipeline_specs[i].num_nodes
                for i in range(0, len(self.pipeline_specs))
            )
            == num_nodes
        )

        list_nodes = list(self.world_info.keys())
        node_index = 0
        pipeline: Optional[OobleckPipeline] = None
        for num_pipeline, pipeline_spec in zip(num_pipelines, self.pipeline_specs):
            for i in range(num_pipeline):
                nodes = list_nodes[node_index : node_index + pipeline_spec.num_nodes]
                node_index += pipeline_spec.num_nodes
                for j in range(self.num_gpus_per_node):
                    ranks = [self.world_info[node][j] for node in nodes]
                    pp_pg = dist.new_group(ranks)
                    if dist.get_rank(pp_pg) >= 0:
                        assert pipeline is None, "Pipeline is already allocated for me."
                        pipeline = OobleckPipeline(
                            pipeline_spec,
                            self.model,
                            self.train_dataloader,
                            pp_pg,
                            self.training_args,
                        )
                # TODO: implement FSDP processgroup

        return pipeline

    def train(self):
        """
        Train my pipeline and synchronize gradients after each schedule is done
        until specified steps or epoch is reached.
        """
        if self.training_args.max_steps > 0:
            for i in range(self.training_args.max_steps):
                logger.info(f"[{i}] step")
                self.my_pipeline.train()
                self.do_allreduce()
                self.my_pipeline.optimizer_step()
        else:
            num_steps = len(self.train_dataloader.loader)

            for _ in range(int(self.training_args.num_train_epochs)):
                for i in range(num_steps):
                    logger.info(f"[{i}] step")
                    self.my_pipeline.train()
                    self.do_allreduce()
                    self.my_pipeline.optimizer_step()


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["NODE_NAME"] = "localhost2"
    os.environ["MAX_NUM_NODES"] = "2"
    os.environ["NUM_GPUS_PER_NODE"] = "1"

    engine = OobleckEngine(0, "gpt2", "wikitext", "wikitext-2-raw-v1")
    engine.init_distributed()
    engine.train()
