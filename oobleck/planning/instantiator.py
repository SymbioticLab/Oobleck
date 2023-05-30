import pyomo.environ as pyomo
import torch.distributed

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from deepspeed import comm as dist
from deepspeed.utils import logger

from pipeline_template import PipelineTemplate, PipelineTemplateGenerator
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.execution.dataloader import OobleckTrainDataLoader
from oobleck.module.model import OobleckModel

from transformers import TrainingArguments


class HeterogeneousPipelineExecutionPlan:
    def __init__(
        self,
        pipeline_templates: List[PipelineTemplate],
        num_instances_set: Dict[PipelineTemplate, int],
        num_microbatches_set: Dict[PipelineTemplate, int],
    ):
        self.pipeline_templates = pipeline_templates
        self.num_instances_set = num_instances_set
        self.num_microbatches_set = num_microbatches_set
        self.num_nodes = sum(
            pipeline_template.get_num_nodes() * num_instances
            for pipeline_template, num_instances in num_instances_set.items()
        )

    def __repr__(self) -> str:
        result = ""
        total_num_microbatches = 0
        for template in self.pipeline_templates:
            if (
                template not in self.num_instances_set
                or template not in self.num_microbatches_set
            ):
                continue
            result += (
                f"{self.num_instances_set[template]} x {template} pipelines "
                f"(num microbatches: {self.num_microbatches_set[template]})\n"
            )
            total_num_microbatches += (
                self.num_instances_set[template] * self.num_microbatches_set[template]
            )
        result += f"total microbatches: {total_num_microbatches}"
        return result

    @property
    def iteration_time(self) -> float:
        # Insu: pipeline templates are all optimal plans.
        # Implement getting iteration time in PipelineTemplate into C++ and use it.
        max_iteration_time = max(
            pipeline_template.get_iteration_time()
            * num_microbatches
            / len(pipeline_template.get_stages())
            for pipeline_template, num_microbatches in self.num_microbatches_set.items()
        )

        # FIXME: currently we only consider communication overhead
        # of the first layer, believing communication of other layers
        # can fully be hidden in backward pass computing.
        synchronization_overhead = (
            Planner()  # Hack: should return instance here... Because Planner is a singleton
            .model_layers[-1]
            .allreduce_cross_nodes[self.num_nodes]
        )

        logger.debug(
            f"iteration_time of execution plan {self.num_instances_set}: {max_iteration_time + synchronization_overhead}"
        )
        return max_iteration_time + synchronization_overhead

    @property
    def average_num_nodes(self) -> float:
        total_num_nodes = sum(
            pipeline_template.get_num_nodes() * num_instance
            for pipeline_template, num_instance in self.num_instances_set.items()
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
        step: int = 0,
    ) -> Tuple[OobleckPipeline, List[List[int]]]:
        my_pipeline: Optional[OobleckPipeline] = None
        total_num_nodes_used = 0

        pipeline_ranks: List[List[int]] = []

        for pipeline_template in self.pipeline_templates:
            for _ in range(self.num_instances_set[pipeline_template]):
                # TODO: implement FSDP by considering spec.num_gpus_per_node
                ranks = list(
                    range(
                        total_num_nodes_used,
                        total_num_nodes_used + pipeline_template.get_num_nodes(),
                    )
                )
                ranks_to_layer_map = [ranks[i] for i in spec.layer_spec]
                pipeline_ranks.append(ranks_to_layer_map)
                total_num_nodes_used += pipeline_template.get_num_nodes()
                process_group = torch.distributed.new_group(ranks)

                logger.info(
                    f"Instantiating a {len(pipeline_template.get_stages())}-stage "
                    f"pipeline with {pipeline_template.get_num_nodes()} nodes"
                )

                if dist.get_rank(process_group) >= 0:
                    assert my_pipeline is None
                    my_pipeline = OobleckPipeline(
                        pipeline_template,
                        model,
                        dataloader,
                        step,
                        process_group,
                        training_args,
                    )

        assert my_pipeline, "No pipeline has been initiated for this rank"
        return my_pipeline, pipeline_ranks


class PipelineInstantiator:
    def get_best_execution_plan(
        self,
        pipeline_templates: List[PipelineTemplate],
        num_nodes: int,
        global_num_microbatch: int,
    ) -> HeterogeneousPipelineExecutionPlan:
        """
        Section 4.2. Pipeline Instantiation implementation
        """
        num_instances_set_list: List[
            Dict[PipelineTemplate, int]
        ] = self._enumerate_instantiation_options(pipeline_templates, num_nodes)

        num_microbatches_set_list: List[Dict[PipelineTemplate, int]] = [
            self._distribute_batch(global_num_microbatch, num_instances_set)
            for num_instances_set in num_instances_set_list
        ]

        execution_plans: List[HeterogeneousPipelineExecutionPlan] = [
            HeterogeneousPipelineExecutionPlan(
                pipeline_templates,
                num_instances_set,
                num_microbatches_set,
            )
            for num_instances_set, num_microbatches_set in zip(
                num_instances_set_list, num_microbatches_set_list
            )
            if num_instances_set is not None and num_microbatches_set is not None
        ]

        result: HeterogeneousPipelineExecutionPlan = min(
            execution_plans, key=lambda plan: plan.iteration_time
        )

        logger.info(f"Best execution plan: {result.num_instances_set}")
        return result

    def _enumerate_instantiation_options(
        self,
        pipeline_templates: List[PipelineTemplate],
        num_nodes: int,
    ) -> List[Dict[PipelineTemplate, int]]:
        """Oobleck paper section 4.2.1. Enumerating instantiation options implementation
        Get all feasible sets of xi's that can use all of the given nodes.
        """

        dp: List[List[List[Dict[PipelineTemplate, int]]]] = [
            [[] for _ in range(num_nodes + 1)]
            for _ in range(len(pipeline_templates) + 1)
        ]

        for i in range(1, len(pipeline_templates) + 1):
            dp[i][0] = [defaultdict(int)]
            for j in range(1, num_nodes + 1):
                # (1) in Figure: copy all dicts
                dp[i][j] = [combo.copy() for combo in dp[i - 1][j]]
                if pipeline_templates[i - 1].num_nodes <= j:
                    # (2) in Figure: copy all dicts with one pipeline_templates[i - 1] added
                    for combo in dp[i][j - pipeline_templates[i - 1].num_nodes]:
                        new_combo = combo.copy()
                        new_combo[pipeline_templates[i - 1]] += 1

                        # concatenate two lists
                        dp[i][j].append(new_combo)

        return dp[-1][-1]

    def _distribute_batch(
        self,
        global_num_microbatch: int,
        num_instances_set: Dict[PipelineTemplate, int],
    ) -> Optional[Dict[PipelineTemplate, int]]:
        """Oobleck paper section 4.2.2. Batch distribution implementation
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
            i: pipeline_template.get_iteration_time() / 10
            for i, pipeline_template in enumerate(num_instances_set)
        }

        x = {
            i: instance_num for i, instance_num in enumerate(num_instances_set.values())
        }
        s = {
            i: len(pipeline_template.get_stages())
            for i, pipeline_template in enumerate(num_instances_set.keys())
        }

        # Define the Pyomo variable
        # nb: number fo microbatches per PipelineSpec
        model.nb = pyomo.Var(model.I, within=pyomo.PositiveIntegers)

        # Objective function
        def objective(model):
            avg_bT = sum(T[i] / s[i] * model.nb[i] for i in model.I) / len(model.I)
            return sum((T[i] / s[i] * model.nb[i] - avg_bT) ** 2 for i in model.I)

        model.obj = pyomo.Objective(rule=objective, sense=pyomo.minimize)

        # Define constraints
        def c1(model):
            return sum(model.nb[i] * x[i] for i in model.I) == global_num_microbatch

        # def c2(model, i):
        #     return model.nb[i] >= 2 * s[i]

        model.constraint1 = pyomo.Constraint(rule=c1)
        # model.constraint2 = pyomo.Constraint(range(len(model.I)), rule=c2)

        pyomo.SolverFactory("mindtpy").solve(
            model, mip_solver="glpk", nlp_solver="ipopt"
        )

        # check for all i model.nb[i].value is integer, otherwise return None
        if not all(model.nb[i].value for i in model.I):
            return None

        nb_optimal = {
            spec: int(model.nb[i].value)
            for i, spec in zip(model.I, num_instances_set.keys())
        }

        logger.debug(f"Number of microbatch per PipelineSpec: {nb_optimal}")
        return nb_optimal
