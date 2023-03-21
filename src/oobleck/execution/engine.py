import os
import math
import torch
import torch.distributed
import pyomo.environ as pyomo

from statistics import mean
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

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
            model_name, self.dataset.sample, training_args, model_args
        )

    def _init_distributed_from_etcd(self):
        pass

    def _init_distributed_from_env(self):
        """Temporary implementation that enables execution
        without master/agent processes.
        Must be removed and replaced with `init_distributed_from_etcd`.
        """
        self.world_info = {
            "localhost0": [0],
            # "localhost1": [1],
            # "localhost2": [2],
            # "localhost3": [3],
        }
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

        # create a list of pipelinespecs that can cover all nodes.
        # this is invariant and never changes over reconfiguration.
        self.pipeline_specs: List[PipelineSpec] = self.create_pipeline_specs(
            self.ft_spec, self.max_num_nodes, self.model
        )

        # num_pipeline_specs = self._get_num_instantiation_pipeline_specs()

        # TODO: move it to user configurable arguments
        global_num_microbatch = 8

        self.instantiate_pipelines(
            self.pipeline_specs, len(self.world_info), global_num_microbatch
        )
        # self.my_pipeline = self._deploy_pipelines(num_pipeline_specs)
        # assert self.my_pipeline, "Cannot find a pipeline that I am belong to."

    def create_pipeline_specs(
        self, ft_spec: int, max_num_nodes: int, model: OobleckModel
    ) -> List[PipelineSpec]:
        """Oobleck paper section 4.1. Configuring PipelineSpecs implementation
        Generates the list of :class:`oobleck.planning.pipeline_spec.PipelineSpec`s
        with heterogeneous number of nodes specifications, a linear combination of
        which can represent any number N (min_num_nodes <= N <= max_num_nodes).

        Oobleck exploits the Frobenius problem and creates a list of `PipelineSpec`s
        based on the constraints:
        1. \# of `PipelineSpec`s > n0 - 1
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
        # min_num_nodes = 2
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
    # Paper section 4.2. is implemented in oobleck.planning.pipeline_spec.PipelineSpec.
    # ==========================================================================================

    def instantiate_pipelines(
        self,
        pipeline_specs: List[PipelineSpec],
        num_nodes: int,
        global_num_microbatch: int,
        throughput_oriented: bool = True,
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
        pipeline_instantiation_set_list: List[
            Dict[PipelineSpec, int]
        ] = self.get_feasible_sets_of_pipeline_instantiation(pipeline_specs, num_nodes)

        num_mb_per_pipelinespec_list: List[Dict[PipelineSpec, int]] = []
        for set in pipeline_instantiation_set_list:
            num_mb_per_pipelinespec_list.append(
                self.distribute_batch(global_num_microbatch, set)
            )

        # For each feasible set, measure throughput
        # FIXME: currently communication time only considers
        # the first layer synchronization, believing communication of
        # other layers can fully be hidden in backward pass computing.
        tmp_throughput: float = math.inf
        tmp_num_avg_nodes: float = math.inf
        tmp_set: Dict[PipelineSpec, Tuple[int, int]] = {}
        total_num_nodes_used = 0

        for num_instance_set, num_mb_per_ppspec in zip(
            pipeline_instantiation_set_list, num_mb_per_pipelinespec_list
        ):

            if throughput_oriented:
                set_throughput = max(
                    [
                        spec.optimal_plan.get_e() * num_mb_per_ppspec[spec]
                        for spec in num_instance_set.keys()
                    ]
                ) + (
                    list(num_instance_set.keys())[0]
                    .planner.model_layers[-1]
                    .allreduce_cross_nodes[num_nodes]
                )
                if tmp_throughput > set_throughput:
                    tmp_throughput = set_throughput
                    tmp_set = {}
                    for spec, num_instance in num_instance_set.items():
                        tmp_set[spec] = (num_instance, num_mb_per_ppspec[spec])
                    total_num_nodes_used += spec.num_nodes * num_instance
            else:
                # average number of nodes instantiated pipelines have
                total_num_nodes = sum(
                    [
                        spec.num_nodes * num_instance
                        for spec, num_instance in num_instance_set.items()
                    ]
                )
                total_instance = sum(list(num_instance_set.values()))
                set_num_avg_nodes = total_num_nodes / total_instance
                if tmp_num_avg_nodes > set_num_avg_nodes:
                    tmp_num_avg_nodes = set_num_avg_nodes
                    tmp_set = {}
                    for spec, num_instance in num_instance_set.items():
                        tmp_set[spec] = tuple(num_instance, num_mb_per_ppspec[spec])
                    total_num_nodes_used += spec.num_nodes * num_instance

        sum_used_nodes = sum(
            [
                spec.num_nodes * num_instance
                for spec, (num_instance, _) in tmp_set.items()
            ]
        )
        assert (
            num_nodes == sum_used_nodes
        ), f"num nodes {num_nodes} is not equal to sum of used nodes {sum_used_nodes}"

        # instantiate pipelines
        total_num_nodes_used = 0
        pipeline_ranks_list: List[List[int]] = []
        pipeline_pgs_list: List[ProcessGroup] = []
        num_total_microbatches = sum(
            num_instance * num_mb for num_instance, num_mb in tmp_set.values()
        )

        for spec, (num_instance, num_mb) in tmp_set.items():
            for i in range(num_instance):
                ranks = list(
                    range(total_num_nodes_used, total_num_nodes_used + spec.num_nodes)
                )
                pipeline_pg = dist.new_group(ranks)
                pipeline_pgs_list.append(pipeline_pg)

                if dist.get_rank(pipeline_pg) >= 0:
                    assert not hasattr(self, "my_pipeline")
                    self.training_args.gradient_accumulation_steps = num_mb

                    self.train_dataloader = OobleckTrainDataLoader(
                        self.dataset.dataset["train"],
                        self.training_args,
                        num_total_microbatches,
                        self.dataset.data_collator,
                    )

                    my_pipeline = OobleckPipeline(
                        spec,
                        self.model,
                        self.train_dataloader,
                        pipeline_pg,
                        self.training_args,
                    )

                ranks_to_layer_map = [ranks[i] for i in spec.layer_spec]
                pipeline_ranks_list.append(ranks_to_layer_map)

                total_num_nodes_used += spec.num_nodes

        assert my_pipeline

        layer_dp_groups: List[ProcessGroup] = []
        for layer_index in range(len(my_pipeline.model_layers)):
            ranks = [ranks[layer_index] for ranks in pipeline_ranks_list]
            dp_pg = dist.new_group(ranks)
            layer_dp_groups.append(dp_pg)

        self.initialize_dp_process_groups(my_pipeline, layer_dp_groups)

        return my_pipeline

    def get_feasible_sets_of_pipeline_instantiation(
        self,
        pipeline_specs: List[PipelineSpec],
        num_nodes: int,
    ) -> List[Dict[PipelineSpec, int]]:
        """Oobleck paper section 4.3.1. Getting number of pipeline instances implementation
        Get all feasible sets of xi's that can use all of the given nodes.

        Args:
            pipeline_specs (List[PipelineSpec]): List of `PipelineSpec`s to be instantiated.
            num_nodes (int): Number of nodes to be used for training.

        Returns:
            List[Dict[PipelineSpec, int]]: List of feasible sets.
                Each set is implemented as a dictionary, key as PipelineSpec and
                value as how many instances should be created on the key PipelineSpec.
        """

        dp: List[List[List[Dict[PipelineSpec, int]]]] = [
            [[] for _ in range(num_nodes + 1)] for _ in range(len(pipeline_specs) + 1)
        ]

        for i in range(1, len(pipeline_specs) + 1):
            dp[i][0] = [defaultdict(int)]
            for j in range(1, num_nodes + 1):
                # (1) in Figure: copy all dicts
                dp[i][j] = [combo.copy() for combo in dp[i - 1][j]]
                if pipeline_specs[i - 1].num_nodes <= j:
                    # (2) in Figure: copy all dicts with one pipeline_specs[i - 1] added
                    for combo in dp[i][j - pipeline_specs[i - 1].num_nodes]:
                        new_combo = combo.copy()
                        new_combo[pipeline_specs[i - 1]] += 1

                        # concatenate two lists
                        dp[i][j].append(new_combo)

        return dp[-1][-1]

    def distribute_batch(
        self,
        global_num_microbatch: int,
        instances_per_pipelinespec: Dict[PipelineSpec, int],
    ) -> Dict[PipelineSpec, int]:
        """Oobleck paper section 4.3.2. Calculating batch size for each pipeline template
        satisfying the following two requirements
        1. std(Bi * Ti) is minimized
        2. sum(Bi) = B
        3. Each Bi is divisible by b (training_args.per_device_train_batch_size)

        Use Pyomo (Python Optimization Modeling Objects).

        Args:
            global_num_microbatch (int): Global batch // training_args.per_device_train_batch_size.
            instances_per_pp_template (List[int]): List of number of pipeline instances per template
            pipeline_iteration_time (List[float]): List of PipelineSpec.e.

        Returns:
            List[int]: List of number of microbatch per template.
                Multiply by training_args.per_device_train_batch_size to calculate minibatch size.
        """

        model = pyomo.ConcreteModel()

        model.I = pyomo.Set(initialize=list(range(len(instances_per_pipelinespec))))
        T = {
            i: pipeline_spec.optimal_plan.get_e()
            for i, pipeline_spec in enumerate(instances_per_pipelinespec)
        }
        x = {
            i: instance_num
            for i, instance_num in enumerate(instances_per_pipelinespec.values())
        }

        # Define the Pyomo variable
        # nb: number fo microbatches per PipelineSpec
        model.nb = pyomo.Var(model.I, within=pyomo.PositiveIntegers)

        # Objective function
        def objective(model):
            avg_bT = sum(model.nb[i] * T[i] for i in model.I) / len(model.I)
            return sum((model.nb[i] * T[i] - avg_bT) ** 2 for i in model.I)

        model.obj = pyomo.Objective(rule=objective)

        # Define constraints
        def c1(model):
            return sum(model.nb[i] * x[i] for i in model.I) == global_num_microbatch

        model.constraint1 = pyomo.Constraint(rule=c1)

        pyomo.SolverFactory("mindtpy").solve(
            model, mip_solver="glpk", nlp_solver="ipopt"
        )

        nb_optimal = {
            spec: int(model.nb[i].value)
            for i, spec in zip(model.I, instances_per_pipelinespec.keys())
        }
        # nb_optimal = [int(model.nb[i].value) for i in model.I]
        logger.info(f"Number of microbatch per PipelineSpec: {nb_optimal}")
        return nb_optimal

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
                try:
                    self.my_pipeline.train()
                except StopIteration:
                    self.my_pipeline.reset_data_iterator()
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
                self.my_pipeline.reset_data_iterator()
