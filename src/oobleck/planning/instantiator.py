import pyomo.environ as pyomo

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from deepspeed import comm as dist
from deepspeed.utils import logger

from oobleck.planning.pipeline_spec import PipelineSpec
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.execution.dataloader import OobleckTrainDataLoader
from oobleck.module.model import OobleckModel

from transformers import TrainingArguments


class HeterogeneousPipelineExecutionPlan:
    def __init__(
        self,
        pipeline_specs: List[PipelineSpec],
        num_instances_set: Dict[PipelineSpec, int],
        num_microbatches_set: Dict[PipelineSpec, int],
    ):
        self.pipeline_specs = pipeline_specs
        self.num_instances_set = num_instances_set
        self.num_microbatches_set = num_microbatches_set
        self.num_nodes = sum(
            spec.num_nodes * num_instances
            for spec, num_instances in num_instances_set.items()
        )

    @property
    def iteration_time(self) -> float:
        max_iteration_time = max(
            spec.optimal_plan.get_e() * num_microbatches
            for spec, num_microbatches in self.num_microbatches_set.items()
        )

        # FIXME: currently we only consider communication overhead
        # of the first layer, believing communication of other layers
        # can fully be hidden in backward pass computing.
        synchronization_overhead = (
            self.pipeline_specs[0]
            .planner.model_layers[-1]
            .allreduce_cross_nodes[self.num_nodes]
        )

        logger.info(
            f"iteration_time of execution plan {self.num_instances_set}: {max_iteration_time + synchronization_overhead}"
        )
        return max_iteration_time + synchronization_overhead

    @property
    def average_num_nodes(self) -> float:
        total_num_nodes = sum(
            spec.num_nodes * num_instance
            for spec, num_instance in self.num_instances_set.items()
        )
        total_num_instances = sum(list(self.num_instances_set.values()))
        return total_num_nodes / total_num_instances

    def get_my_number_of_microbatches(self, global_rank: int) -> int:
        """This is for creating dataloader with number of microbatches.

        Returns:
            int: number of microbatches for this rank
        """
        num_ranks_used = 0
        for spec in self.pipeline_specs:
            num_ranks_spec = (
                spec.num_nodes * spec.num_gpus_per_node * self.num_instances_set[spec]
            )
            if global_rank in range(num_ranks_used, num_ranks_used + num_ranks_spec):
                return self.num_microbatches_set[spec]
            num_ranks_used += num_ranks_spec

        ValueError("Cannot find a range that the global rank falls.")

    def instantiate(
        self,
        model: OobleckModel,
        dataloader: OobleckTrainDataLoader,
        training_args: TrainingArguments,
    ) -> Tuple[OobleckPipeline, List[List[int]]]:
        my_pipeline: Optional[OobleckPipeline] = None
        total_num_nodes_used = 0

        pipeline_ranks: List[List[int]] = []

        for spec in self.pipeline_specs:
            for i in range(self.num_instances_set[spec]):
                # TODO: implement FSDP by considering spec.num_gpus_per_node
                ranks = list(
                    range(total_num_nodes_used, total_num_nodes_used + spec.num_nodes)
                )
                ranks_to_layer_map = [ranks[i] for i in spec.layer_spec]
                pipeline_ranks.append(ranks_to_layer_map)
                total_num_nodes_used += spec.num_nodes
                process_group = dist.new_group(ranks)

                if dist.get_rank(process_group) >= 0:
                    assert my_pipeline is None
                    my_pipeline = OobleckPipeline(
                        i, spec, model, dataloader, process_group, training_args
                    )

        assert my_pipeline, "No pipeline has been initiated for this rank"
        return my_pipeline, pipeline_ranks


class PipelineInstantiator:
    def __init__(
        self,
        pipeline_specs: List[PipelineSpec],
        num_nodes: int,
        global_num_microbatch: int,
    ):
        """Oobleck paper section 4.3. Instantiating Pipeline Templates implementation
        Instantiate given `PipelineSpec`s and create `OobleckPipeline`s.

        Args:
            pipeline_specs (List[PipelineSpec]): List of `PipelineSpec`s to be
                used for instantiation.
            num_nodes: int: Number of nodes.
            throughput_oriented (bool, optional): Whether throughput oriented or
                reconfiguration overhead oriented.
        """
        self.pipeline_specs = pipeline_specs
        self.num_nodes = num_nodes
        self.global_num_microbatch = global_num_microbatch

        num_instances_set_list: List[
            Dict[PipelineSpec, int]
        ] = self._get_feasible_sets_of_pipeline_instantiation(pipeline_specs, num_nodes)

        # For each feasible xi set, calculate batch distribution and stores
        # a tuple of number of instance of the pipelinespec, and batch size.
        num_microbatches_set_list: List[Dict[PipelineSpec, int]] = [
            self._distribute_batch(self.global_num_microbatch, num_instances_set)
            for num_instances_set in num_instances_set_list
        ]

        self.execution_plans: List[HeterogeneousPipelineExecutionPlan] = [
            HeterogeneousPipelineExecutionPlan(
                self.pipeline_specs,
                num_instances_set,
                num_microbatches_set,
            )
            for spec, num_instances_set, num_microbatches_set in zip(
                self.pipeline_specs, num_instances_set_list, num_microbatches_set_list
            )
        ]

    def get_best_execution_plan(
        self, throughput_oriented: bool = True
    ) -> HeterogeneousPipelineExecutionPlan:
        result: HeterogeneousPipelineExecutionPlan = None
        if throughput_oriented:
            result = min(self.execution_plans, key=lambda plan: plan.iteration_time)
        else:
            result = max(self.execution_plans, key=lambda plan: plan.average_num_nodes)

        logger.info(f"Best execution plan: {result.num_instances_set}")
        return result

    def _get_feasible_sets_of_pipeline_instantiation(
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

    def _distribute_batch(
        self,
        global_num_microbatch: int,
        num_instances_set: Dict[PipelineSpec, int],
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

        model.I = pyomo.Set(initialize=list(range(len(num_instances_set))))
        T = {
            i: (
                pipeline_spec.optimal_plan.get_e()
                / (4 * len(pipeline_spec.optimal_plan.stages))
                * 1_000_000
            )
            for i, pipeline_spec in enumerate(num_instances_set)
        }
        x = {
            i: instance_num for i, instance_num in enumerate(num_instances_set.values())
        }

        # Define the Pyomo variable
        # nb: number fo microbatches per PipelineSpec
        model.nb = pyomo.Var(model.I, within=pyomo.PositiveIntegers)

        # Objective function
        def objective(model):
            avg_bT = sum(model.nb[i] * T[i] for i in model.I) / len(model.I)
            return sum((model.nb[i] * T[i] - avg_bT) ** 2 for i in model.I)

        model.obj = pyomo.Objective(rule=objective, sense=pyomo.minimize)

        # Define constraints
        def c1(model):
            return sum(model.nb[i] * x[i] for i in model.I) == global_num_microbatch

        model.constraint1 = pyomo.Constraint(rule=c1)

        pyomo.SolverFactory("mindtpy").solve(
            model, mip_solver="glpk", nlp_solver="ipopt"
        )

        nb_optimal = {
            spec: int(model.nb[i].value)
            for i, spec in zip(model.I, num_instances_set.keys())
        }

        logger.info(f"Number of microbatch per PipelineSpec: {nb_optimal}")
        return nb_optimal
