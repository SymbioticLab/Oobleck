import os
import math
import torch
import torch.distributed

from collections import defaultdict
from typing import Optional, Dict, List, Any

from torch.distributed import ProcessGroup

from deepspeed import comm as dist
from deepspeed.utils import logger

from oobleck.execution.dataset import OobleckDataset
from oobleck.module.model import OobleckModel
from oobleck.planning.pipeline_spec import PipelineSpec
from oobleck.elastic.client import ElasticWorkerClientMixin, ElasticClientMonitorMixin
from oobleck.execution.dataloader import OobleckTrainDataLoader
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.utils.timer import OobleckTimer, measure_time

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
    ElasticClientMonitorMixin,
    ElasticWorkerClientMixin,
    DataSynchronizationMixin,
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
        training_args = TrainingArguments("/tmp/output", gradient_accumulation_steps=4)
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

        # create a list of pipelinespecs that can cover all nodes.
        # this is invariant and never changes over reconfiguration.
        self.pipeline_specs: List[PipelineSpec] = self.create_pipeline_specs(
            self.ft_spec, self.max_num_nodes, self.model
        )

    def _init_distributed_from_etcd(self):
        pass

    def _init_distributed_from_env(self):
        """Temporary implementation that enables execution
        without master/agent processes.
        Must be removed and replaced with `init_distributed_from_etcd`.
        """
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
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "25400"

        dist.init_distributed("nccl", auto_mpi_discovery=False)

    def init_distributed(self):
        # TODO: setup with etcd
        self._init_distributed_from_env()
        self.timer = OobleckTimer()

        self.num_pipeline_specs = self._get_num_instantiation_pipeline_specs()

        # TODO: change total number of microbatches properly.
        self.train_dataloader = OobleckTrainDataLoader(
            self.dataset.dataset["train"],
            self.training_args,
            self.training_args.gradient_accumulation_steps
            * sum([num_spec for num_spec in self.num_pipeline_specs]),
            self.dataset.data_collator,
        )

        self.my_pipeline = self._deploy_pipelines()
        assert self.my_pipeline, "Cannot find a pipeline that I am belong to."

    def _get_num_instantiation_pipeline_specs(self) -> List[int]:
        num_nodes = dist.get_world_size() // self.num_gpus_per_node
        assert num_nodes <= self.max_num_nodes, "World size is larger than max # nodes."

        # Current Alpha-level implementation: always prefer smaller pipelines.
        # TODO: analyze the best optimal combination that has the highest throughput.
        num_pipeline_specs = [0] * len(self.pipeline_specs)
        num_pipeline_specs[0] = math.floor(num_nodes / self.pipeline_specs[0].num_nodes)
        total_assigned_nodes = num_pipeline_specs[0] * self.pipeline_specs[0].num_nodes
        assert (
            total_assigned_nodes <= num_nodes
        ), f"total assigned nodes {total_assigned_nodes} is not less than total given nodes {num_nodes}"

        smallest_non_zero_pipeline_index = 0
        while total_assigned_nodes < num_nodes:
            while (
                smallest_non_zero_pipeline_index < len(self.pipeline_specs)
                and num_pipeline_specs[smallest_non_zero_pipeline_index] == 0
            ):
                smallest_non_zero_pipeline_index += 1

            if (
                smallest_non_zero_pipeline_index + 1 < len(self.pipeline_specs)
                and num_pipeline_specs[smallest_non_zero_pipeline_index] > 0
            ):
                num_pipeline_specs[smallest_non_zero_pipeline_index] -= 1
                num_pipeline_specs[smallest_non_zero_pipeline_index + 1] += 1
                total_assigned_nodes += 1

        assert (
            sum(
                num_pipeline_specs[i] * self.pipeline_specs[i].num_nodes
                for i in range(0, len(self.pipeline_specs))
            )
            == num_nodes
        )

        return num_pipeline_specs

    def _deploy_pipelines(self) -> OobleckPipeline:
        """Calculate required number of heterogeneous pipelines that
        a linear combination of the pipelines fills the distributed world.
        """

        list_nodes = list(self.world_info.keys())
        node_index = 0
        pipeline_process_group_ranks: List[List[int]] = []
        my_pipeline: Optional[OobleckPipeline] = None
        for num_pipeline, pipeline_spec in zip(
            self.num_pipeline_specs, self.pipeline_specs
        ):
            for i in range(num_pipeline):
                nodes = list_nodes[node_index : node_index + pipeline_spec.num_nodes]
                node_index += pipeline_spec.num_nodes
                for j in range(self.num_gpus_per_node):
                    ranks = [self.world_info[node][j] for node in nodes]
                    pp_pg = dist.new_group(ranks)
                    if dist.get_rank(pp_pg) >= 0:
                        my_pipeline = OobleckPipeline(
                            pipeline_spec,
                            self.model,
                            self.train_dataloader,
                            pp_pg,
                            self.training_args,
                        )

                    # Ranks per each layer. Used for creating :class:`ProcessGroup`s for each layer.
                    ranks_to_layer_map = [ranks[i] for i in pipeline_spec.optimal_plan]
                    pipeline_process_group_ranks.append(ranks_to_layer_map)

                # TODO: implement FSDP processgroup.
                # Currently self.world_info[node] always has one rank,
                # all ranks in self.world_info[node] are for FSDP.

        # For now, we don't use FSDP, thus pipelines are not sharded.
        # Generate PGs for each layers in every pipelines.
        # TODO: later when we adopt FSDP, each sharded point must be distinguished.
        layer_dp_groups: List[ProcessGroup] = []
        for layer_index in range(len(my_pipeline.model_layers)):
            ranks = [ranks[layer_index] for ranks in pipeline_process_group_ranks]
            dp_pg = dist.new_group(ranks)
            layer_dp_groups.append(dp_pg)

        self.initialize_dp_process_groups(my_pipeline, layer_dp_groups)

        return my_pipeline

    def create_pipeline_specs(
        self, ft_spec: int, max_num_nodes: int, model: OobleckModel
    ) -> List[PipelineSpec]:
        """Oobleck paper section 4.1.1. Configuring PipelineSpecs implementation
        Generates the list of :class:`oobleck.planning.pipeline_spec.PipelineSpec`s
        with heterogeneous number of nodes specifications, a linear combination of
        which can represent any number N (min_num_nodes <= N <= max_num_nodes).

        Oobleck exploits the Frobenius problem and creates a list of `PipelineSpec`s
        based on the constraints:
        1. # of `PipelineSpec`s > n0 - 1
        2. ni's are consecutive integers (ni + 1 = n(i+1))
        We define n0 = minimum number of nodes ths is required to hold states for training.

        Args:
            ft_spec (int): Fault tolerant spec.
                Oobleck tries to create at least ft_spec + 1 model replica.
            max_num_nodes (int): Maximum # nodes in the cluster.

        Returns:
            List[PipelineSpec]: List of `PipelineSpec`s.
                Length of the list is the number of `PipelineSpec`s, p.
        """
        assert ft_spec >= 0, "Fault tolerance spec must not be negative."

        required_memory = model.total_num_params * 12
        required_min_gpus = math.ceil(
            required_memory / torch.cuda.get_device_properties("cuda:0").total_memory
        )
        min_num_nodes = math.ceil(required_min_gpus / self.num_gpus_per_node)
        assert (
            ft_spec + 1
        ) * min_num_nodes <= max_num_nodes, f"Maximum # nodes ({max_num_nodes}) cannot be smaller than minimum # nodes ({min_num_nodes})."
        if (ft_spec + 1) * min_num_nodes > max_num_nodes:
            logger.warning(
                "The number of nodes is not enough to provide at least ft_spec + 1 copy of the model."
                "Oobleck may fail to provide fault tolerancy if continue."
            )

        # p = n0 - 1
        num_pipeline_specs = min_num_nodes
        if num_pipeline_specs < 1:
            num_pipeline_specs = 1

        pipeline_specs = list(range(min_num_nodes, min_num_nodes + num_pipeline_specs))
        assert all(
            num_nodes <= max_num_nodes for num_nodes in pipeline_specs
        ), "Some PipelineSpec needs to have more # nodes than maximum # nodes (impossible)."

        return [
            PipelineSpec(num_nodes, self.num_gpus_per_node, self.model)
            for num_nodes in pipeline_specs
        ]

    # ==========================================================================================
    # Paper section 4.1.2. is implemented in oobleck.planning.pipeline_spec.PipelineSpec.
    # ==========================================================================================

    def instantiate_pipelines(
        self,
        pipeline_specs: List[PipelineSpec],
        num_nodes: int,
        throughput_oriented: bool = True,
    ) -> List[List[OobleckPipeline]]:
        """Oobleck paper section 4.1.3. Instantiating PipelineSpecs implementation
        Instantiates given `PipelineSpec`s to create actual :class:`oobleck.execution.pipeline.Pipeline`s.

        Oobleck uses dynamic programming to find all feasible solutions, then picks one
        in accordance with the user input `throughput_oriented`.

        TODO: change it to degree factor, not boolean.

        Args:
            pipeline_specs (List[PipelineSpec]): List of `PipelineSpec`s to be instantiated.
            num_nodes (int): Number of nodes to be used for training.
            throughput_oriented (bool): Whether instantiation should priortize throughput or
            smaller reconfiguration overhead.

        Returns:
            List[List[OobleckPipeline]]: List of instantiated `OobleckPipeline`s.
                Inner `List[OobleckPipeline]` includes the list of instantiated pipelines with a single
                `PipelineSpec`.
        """

        def dp_get_all_solutions() -> List[Dict[int, int]]:
            dp: List[List[List[Dict[int, int]]]] = [
                [[] for _ in range(num_nodes + 1)]
                for _ in range(len(pipeline_specs) + 1)
            ]

            for i in range(1, len(pipeline_specs) + 1):
                dp[i][0] = [defaultdict(int)]
                for j in range(1, num_nodes + 1):
                    # (1) in Figure: copy all dicts
                    dp[i][j] = [combo.copy() for combo in dp[i - 1][j]]
                    if pipeline_specs[i - 1].num_nodes <= j:
                        # (2) in Figure: copy all dicts with one pipeline_specs[i - 1] added
                        for combo in dp[i][j - pipeline_specs[i - 1]]:
                            new_combo = combo.copy()
                            new_combo[pipeline_specs[i - 1]] += 1

                            # concatenate two lists
                            dp[i][j].append(new_combo)

            return dp[-1][-1]

        solutions: List[Dict[int, int]] = dp_get_all_solutions()
        # measure throughput of each solution using profiled data
        assert False, "Finish implementation"

    def train(self):
        """
        Train my pipeline and synchronize gradients after each schedule is done
        until specified steps or epoch is reached.
        """

        def log():
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

        if self.training_args.max_steps > 0:
            for i in range(self.training_args.max_steps):
                logger.info(f"[{i}] step")
                self.my_pipeline.train()
                self.do_allreduce()
                self.my_pipeline.optimizer_step()
                log()
        else:
            num_steps = len(self.train_dataloader)

            for _ in range(int(self.training_args.num_train_epochs)):
                for i in range(num_steps):
                    logger.info(f"[{i}] step")
                    self.my_pipeline.train()
                    self.do_allreduce()
                    self.my_pipeline.optimizer_step()
                    log()
