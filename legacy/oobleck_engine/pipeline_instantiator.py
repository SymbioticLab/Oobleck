from collections import defaultdict

import pulp
from cornstarch.pipeline_template import PipelineTemplate
from loguru import logger


class PipelineInstantiator:
    """A class that determines the number of pipelines to be instantiated
    from each pipeline template and the number of microbatches.
    """

    def __init__(
        self,
        pipeline_templates: dict[int, PipelineTemplate],
        global_num_microbatches: int,
        fault_tolerance_threshold: int,
    ):
        self.pipeline_templates = pipeline_templates
        self.global_num_microbatches = global_num_microbatches
        self.fault_tolerance_threshold = fault_tolerance_threshold

    def instantiate(
        self, num_nodes: int
    ) -> tuple[dict[PipelineTemplate, int], dict[PipelineTemplate, int]]:
        """Instantiate pipelines from pipeline templates.
        If there are already instantiated pipelines, this method will reconfigure
        pipelines for a new distributed configuration.

        Args:
            num_nodes (int): The number of nodes in the distributed environment.

        Returns:
            A tuple of two dict[PipelineTemplate, int] objects, where
            1. dict[PipelineTemplate, int]: the number of pipelines to be instantiated
            2. dict[PipelineTemplate, int]: the number of microbatches per pipeline

        Raises:
            RuntimeError:
                1. If no optimal batch distribution found for the given number of nodes
                   and the global batch size.
                2. If failed to find a set of pipeline templates for the given number of nodes.
        """
        instantiations_options = self._enumerate_instantiation_options(num_nodes)

        # Call self._distribute_batch for each element in instantiations_options
        # Should also include None to properly calculate "index" of optimal dist
        batch_distributions = [
            self.distribute_batch(option) for option in instantiations_options
        ]

        if all(dist is None for dist in batch_distributions):
            raise RuntimeError(
                f"Failed to find optimal batch distribution for {num_nodes} nodes."
            )

        str = "Batch distributions===============\n"
        for latency_dist in batch_distributions:
            if latency_dist is None:
                continue
            latency, dist = latency_dist
            str += f"  {dist} (latency {latency} ms)\n"
        str + "=================================="
        logger.debug(str)

        # Find the second dictionary where its corresponding float is the minimum
        optimal_distribution = min(
            [dist for dist in batch_distributions if dist is not None],
            key=lambda x: x[0],
        )
        index = batch_distributions.index(optimal_distribution)

        logger.info(f"Optimal batch distribution: {optimal_distribution[1]}")

        return (
            instantiations_options[index],
            optimal_distribution[1],
        )

    def _enumerate_instantiation_options(
        self, num_nodes: int
    ) -> list[dict[PipelineTemplate, int]]:
        """Use dynamic programming to get all feasible sets of pipeline templates
        given number of nodes.
        Implementation of Section 4.2.1.

        Args:
            num_nodes (int): The number of nodes in the distributed environment.

        Returns:
            A list of dict[PipelineTemplate, int] objects, where each dict represents
            a feasible set of pipeline templates.
        """
        logger.debug(
            f"Enumerating all feasible sets of pipeline templates for {num_nodes} nodes."
        )

        pipeline_templates: list[PipelineTemplate] = list(
            self.pipeline_templates.values()
        )

        dp: list[list[list[dict[PipelineTemplate, int]]]] = [
            [[] for _ in range(num_nodes + 1)]
            for _ in range(len(self.pipeline_templates) + 1)
        ]

        for i in range(1, len(pipeline_templates) + 1):
            dp[i][0] = [defaultdict(int)]
            for j in range(1, num_nodes + 1):
                # (1) in Figure: copy all dicts
                dp[i][j] = [combo.copy() for combo in dp[i - 1][j]]
                if pipeline_templates[i - 1].num_stages <= j:
                    # (2) in Figure: copy all dicts with one pipeline_templates[i - 1] added
                    for combo in dp[i][j - pipeline_templates[i - 1].num_stages]:
                        new_combo = combo.copy()
                        new_combo[pipeline_templates[i - 1]] += 1

                        # concatenate two lists
                        dp[i][j].append(new_combo)

        logger.debug(f"Dynamic programming result: {dp[-1][-1]}")
        if not dp[-1][-1]:
            raise RuntimeError(
                f"Failed to find feasible sets of pipeline templates for {num_nodes} nodes."
            )
        return [
            option
            for option in dp[-1][-1]
            if sum(option.values()) >= self.fault_tolerance_threshold
        ]

    def distribute_batch(
        self,
        num_pipelines: dict[PipelineTemplate, int],
        need_all_pipelines_have_batch: bool = False,
    ) -> tuple[float, dict[PipelineTemplate, int]] | None:
        """Find the optimal distribution of microbatches that minimizes iteration time.
        Implementation of Section 4.2.2.

        Use integer linear programming library (PuLP) that finds the optimal solution:
        - Let Z the slowest iteration time (max(Bi * Ti))
        - Minimize Z, while keeping sum(Bi) remain constant to global_num_microbatch

        Args:
            global_num_microbatch (int): The total number of microbatches.
            num_pipelines (dict[PipelineTemplate, int]): A set of pipeline templates,
                where each value represents the number of pipelines to be instantiated.

        Returns:
            A tuple of two objects:
                1. float: The optimal iteration time.
                2. A dict[PipelineTemplate, int] object, where each key represents
                   a pipeline template and each value represents the number of microbatches
                   for that pipeline template.

            Or None if the optimal solution is not found.
        """

        assert sum(num_pipelines.values()) >= self.fault_tolerance_threshold, (
            f"The number of pipelines {sum(num_pipelines.values())} should be "
            "greater than or equal to the fault tolerance threshold "
            f"{self.fault_tolerance_threshold}."
        )

        model = pulp.LpProblem("Microbatch Distribution", pulp.LpMinimize)

        # define variables
        num_microbatches = pulp.LpVariable.dicts(
            "num_microbatches",
            num_pipelines.keys(),
            lowBound=1 if need_all_pipelines_have_batch else 0,
            cat=pulp.LpInteger,
        )
        global_iteration_time = pulp.LpVariable("Z", lowBound=0, cat=pulp.LpContinuous)

        # define constraints
        model += (
            sum(
                num_microbatches[template] * num_pipelines[template]
                for template in num_microbatches
            )
            == self.global_num_microbatches
        )
        for template in num_microbatches.keys():
            model += global_iteration_time >= template.latency(
                num_microbatches[template]
            )

        # define objective function
        model += global_iteration_time
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[model.status] != "Optimal":
            logger.warning(f"Failed to find optimal solution for {num_pipelines}.")
            return None

        assert (
            float(global_iteration_time.value()) is not None
        ), "Status is optimal but global_iteration_time is None."

        num_microbatches = {
            template: int(num_microbatch.value())
            for template, num_microbatch in num_microbatches.items()
        }
        logger.debug(
            f"Optiomal batch distribution for {num_pipelines}: {num_microbatches}"
        )
        return (float(global_iteration_time.value()), num_microbatches)
