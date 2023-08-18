from __future__ import annotations

import copy
import math
import socket
import threading
import weakref
from collections import defaultdict
from multiprocessing import connection

import deepspeed.comm as dist
import torch.distributed
from deepspeed.utils.logging import LoggerFactory
from transformers.training_args import TrainingArguments as HFTrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
    get_profile_results,
)
from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader, OobleckSampler
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)

logger = LoggerFactory.create_logger("oobleck_engine")


class ReconfigurationEngine:
    def __init__(self, engine: OobleckEngine, pipelines: list[OobleckPipeline]):
        self._engine = weakref.ref(engine)
        self._pipelines = pipelines
        self._num_instances_set: dict[PipelineTemplate, int] = defaultdict(int)
        for pipeline in self._pipelines:
            self._num_instances_set[pipeline._template] += 1
        self._min_num_ranks = (
            engine._pipeline_templates[0]._num_nodes
            * engine._pipeline_templates[0]._num_gpus_per_node
        )
        self._reconfiguration_listener = threading.Thread(
            target=self._reconfiguration_listener_fn, daemon=True
        )
        self._reconfiguration_listener.start()

    @property
    def engine(self):
        return self._engine()

    def _reconfiguration_listener_fn(self):
        while True:
            self._on_receive_reconfiguration_notification()

    def _on_receive_reconfiguration_notification(self):
        """A method that will be executed in a separate thread
        Waiting for an event from the agent to reconfigure the pipeline.

        Once receive a signal from the agent, the worker will:
        1. destroy current process group
        2. reconfigure the pipeline with lost rank information
        """
        try:
            engine = self.engine
            lost_node: str = engine._agent_pipe.recv()
            lost_ranks = self.remove_lost_node_from_dist_info(lost_node)

            engine.initialize_distributed()
            self.on_reconfigure(lost_ranks)
        except (EOFError, ValueError):
            # Connection closed. Exit.
            pass

    def remove_lost_node_from_dist_info(self, lost_node_ip: str) -> list[int]:
        engine = self.engine
        assert (
            hasattr(engine, "_dist_info") and engine._dist_info is not None
        ), "Distributed is not initialized yet."
        engine._dist_info.agent_ips.remove(lost_node_ip)
        engine._dist_info.world_size -= engine._num_gpus_per_node
        return engine._rank_map.pop(lost_node_ip)

    def on_reconfigure(self, lost_ranks: list[int]):
        def get_pipeline_template(
            ranks: list[int], pipeline_templates: list[PipelineTemplate]
        ) -> PipelineTemplate | None:
            return next(
                (
                    template
                    for template in pipeline_templates
                    if template._num_nodes * template._num_gpus_per_node == len(ranks)
                ),
                None,
            )

        # Copy existing ranks list to use it for data copy
        # layer index -> list of ranks
        old_rank_grids: list[dict[int, list[int]]] = [
            copy.deepcopy(pipeline.rank_grid) for pipeline in self._pipelines
        ]

        # Update ranks first
        for pipeline in self._pipelines:
            pipeline._ranks = [
                rank for rank in pipeline._ranks if rank not in lost_ranks
            ]

        need_merge: bool = False
        new_ranks_list: list[list[int]] = []
        # Prepare new instances set.
        for pipeline in self._pipelines:
            ranks = pipeline._ranks

            # If all ranks are gone, skip it.
            if len(ranks) == 0:
                continue

            # If there is an available template, use it.
            if len(ranks) >= self._min_num_ranks:
                new_ranks_list.append(ranks)
                continue

            # This pipeline needs more ranks
            biggest_pipeline: OobleckPipeline = None
            while len(ranks) < self._min_num_ranks:
                biggest_pipeline = self._find_biggest_pipeline(self._pipelines)
                if biggest_pipeline is None:
                    # No pipelines can yield a rank. Simply add it and handle later.
                    need_merge = True
                    break

                while (
                    len(biggest_pipeline._ranks) > self._min_num_ranks
                    and len(ranks) < self._min_num_ranks
                ):
                    ranks.append(biggest_pipeline._ranks.pop())

            new_ranks_list.append(ranks)

        # Merge pipelines if needed
        if need_merge:
            new_ranks_list = self._merge_pipelines(new_ranks_list)

        # sort ranks for each list of ranks
        for ranks in new_ranks_list:
            ranks.sort()

        # Sort ranks by length so that smaller pipeline ranks always come first.
        # For pipelines with the same number of ranks, a pipeline with smaller rank id comes first.
        new_ranks_list.sort(key=lambda ranks: (len(ranks), ranks[0]))

        # Creae new instances set
        new_num_instances_set: dict[PipelineTemplate, int] = defaultdict(int)
        for ranks in new_ranks_list:
            template = get_pipeline_template(ranks, self.engine._pipeline_templates)
            new_num_instances_set[template] += 1

        new_pipeline = self._reinstantiate(new_num_instances_set, new_ranks_list)

        # Copy model states here
        new_rank_grids: list[dict[int, list[int]]] = []
        for pipeline_template, num_instance in new_num_instances_set.items():
            for _ in range(num_instance):
                rank_grid = pipeline_template.get_rank_grid(new_ranks_list.pop(0))
                new_rank_grids.append(rank_grid)
        self._copy_model_states(old_rank_grids, new_rank_grids, new_pipeline)

        # Before deleting the old pipeline, remove all GPU tensors
        for layer in self.engine._pipeline.execution._layers:
            layer.remove_tensors()

        self.engine._pipeline = new_pipeline

    def _reinstantiate(
        self,
        num_instances_set: dict[PipelineTemplate, int],
        new_ranks_list: list[list[int]],
    ) -> OobleckPipeline:
        global_num_microbatch = (
            self.engine._args.global_microbatch_size
            // self.engine._args.microbatch_size
        )
        instantiator = PipelineInstantiator()
        execution_plan: HeterogeneousPipelinesExecutionPlan = (
            instantiator.get_new_execution_plan(
                new_num_instances_set=num_instances_set,
                allreduce_across_nodes=[
                    layer._allreduce_across_nodes
                    for layer in self.engine._profile_results.get()
                ],
                global_num_microbatch=global_num_microbatch,
            )
        )

        existing_sampler: OobleckSampler = (
            self.engine._pipeline._dataloader.batch_sampler
        )
        new_dataloader: OobleckDataLoader = OobleckDataLoader(
            args=self.engine._hf_training_args,
            datasets=self.engine._dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index=execution_plan.my_pipeline_index,
            num_microbatches=execution_plan.num_microbatches,
            num_iterations_done=existing_sampler.num_iterations_done,
            epoch=existing_sampler.epoch,
        )

        (
            new_pipeline,
            pipelines,
            process_groups_dp,
        ) = execution_plan.instantiate(
            model=self.engine._model,
            dataloader=new_dataloader,
            training_args=self.engine._hf_training_args,
            num_gpus_per_node=self.engine._num_gpus_per_node,
            ranks=new_ranks_list,
            step=self.engine._pipeline._global_step,
        )

        for pipeline in pipelines:
            pipeline.initialize_distributed_fsdp()
            pipeline.initialize_distributed_pipeline()
        new_pipeline.initialize_execution(self.engine._model, self.engine._pipeline)

        self.engine._dp_engine = DataParallelEngine(self.engine, process_groups_dp)
        self._pipelines = pipelines

        return new_pipeline

    def _copy_model_states(
        self,
        old_rank_grids: list[dict[int, list[int]]],
        new_rank_grids: list[dict[int, list[int]]],
        new_pipeline: OobleckPipeline,
    ):
        """
        Copy missing model states in the GPU due to reconfiguration into self.engine._model.
        Then remove unused tensors from the GPU.
        """

        # Iterate all layers to copy model states
        for layer_index in range(len(old_rank_grids[0])):
            old_ranks: list[list[int]] = [
                ranks[layer_index] for ranks in old_rank_grids
            ]
            new_ranks: list[list[int]] = [
                ranks[layer_index] for ranks in new_rank_grids
            ]

            # Check if it is not necessary to copy data.
            if all(rank in old_ranks for rank in new_ranks):
                continue

            # One of the following sets of ranks will send data.
            alive_ranks_in_layer: list[list[int]] = [
                ranks for ranks in old_ranks if ranks in new_ranks
            ]
            if not alive_ranks_in_layer:
                raise RuntimeError(
                    f"No alive ranks for the layer {layer_index}. Terminating."
                )

            # Pick any set of ranks to send model states.
            # TODO: optimize data communication
            ranks_to_send: list[int] = alive_ranks_in_layer[0]

            my_rank = dist.get_rank()
            for ranks_recv in new_ranks:
                if my_rank in ranks_recv:
                    fsdp_index = ranks_recv.index(my_rank)
                    dp_group = self.engine._dp_engine._dp_process_groups[layer_index][
                        fsdp_index
                    ]

                    if my_rank == ranks_to_send[fsdp_index]:
                        param = next(
                            layer
                            for layer in self.engine._pipeline.execution._layers
                            if layer.layer_id == layer_index
                        )._param_handle.flat_param
                        next(
                            layer
                            for layer in new_pipeline.execution._layers
                            if layer.layer_id == layer_index
                        )._param_handle.flat_param.data = param.data
                    else:
                        param = next(
                            layer
                            for layer in new_pipeline.execution._layers
                            if layer.layer_id == layer_index
                        )._param_handle.flat_param

                    dist.broadcast(
                        tensor=param,
                        src=ranks_to_send[fsdp_index],
                        group=dp_group,
                        async_op=True,
                    )

        dist.barrier()
        torch.cuda.synchronize()

    def _merge_pipelines(self, ranks_list: list[list[int]]) -> list[list[int]]:
        """
        When this method is called, all pipelines cannot yield a rank
        but still there is a pipeline that needs more ranks.
        Solve this problem by merging at least two pipelines.

        Return: list of ranks for a merged pipeline and remaining pipelines.
        """
        ranks_to_merge: list[list[int]] = []
        results: list[list[int]] = []
        for ranks in ranks_list:
            ranks_to_merge.append(ranks) if len(
                ranks
            ) < self._min_num_ranks else results.append(ranks)

        try:
            # Merge pipelines
            while ranks_to_merge:
                ranks = ranks_to_merge.pop(0)
                try:
                    while len(ranks) < self._min_num_ranks:
                        ranks.extend(ranks_to_merge.pop(0))
                except IndexError:
                    # No more ranks to merge.
                    # Get ranks from result pipeline
                    ranks.extend(results.pop(0))

                assert len(ranks) >= self._min_num_ranks
                results.append(ranks)
        except IndexError:
            raise RuntimeError("Ranks are insufficient")

        assert ranks_to_merge == []
        return results

    def _find_biggest_pipeline(
        self, pipelines: list[OobleckPipeline]
    ) -> OobleckPipeline | None:
        biggest_pipeline: OobleckPipeline | None = None
        for pipeline in pipelines:
            if biggest_pipeline is None or len(pipeline._ranks) >= len(
                biggest_pipeline._ranks
            ):
                biggest_pipeline = pipeline

        # Check if this pipeline can yield a node
        if biggest_pipeline and len(biggest_pipeline._ranks) > self._min_num_ranks:
            return biggest_pipeline

        return None


class DataParallelEngine:
    def __init__(
        self,
        engine: OobleckEngine,
        pipelines: list[OobleckPipeline],
    ):
        self._engine = weakref.ref(engine)

        # Create process groups for data parallelism
        # 2D grid of ranks involved in each layer, sharded by fsdp
        # layer_index -> dict of (fsdp_index -> list of ranks)
        ranks_grid: dict[int, dict[int, list[int]]] = defaultdict(dict)

        for pipeline in pipelines:
            for layer_index, ranks in pipeline.rank_grid.items():
                assert (
                    isinstance(ranks, list) and len(ranks) == engine._num_gpus_per_node
                )
                for fsdp_index, rank in enumerate(ranks):
                    if fsdp_index not in ranks_grid[layer_index]:
                        ranks_grid[layer_index][fsdp_index] = []
                    ranks_grid[layer_index][fsdp_index].append(rank)

        # Create process groups for data parallelism
        dp_process_groups: dict[int, dict[int, dist.ProcessGroup]] = defaultdict(dict)
        fsdp_indices: list[list[int]] = defaultdict(list)
        my_rank = dist.get_rank()
        for layer_index, ranks_per_layer in ranks_grid.items():
            for fsdp_index, ranks in ranks_per_layer.items():
                dp_process_groups[layer_index][fsdp_index] = dist.new_group(ranks)

                if my_rank in ranks:
                    fsdp_indices[layer_index].append(fsdp_index)

        self._dp_process_groups = dp_process_groups
        self._fsdp_indices = fsdp_indices

    @property
    def engine(self):
        return self._engine()

    def do_allreduce(self):
        for layer in self.engine._pipeline.execution._layers:
            process_groups = {
                fsdp_index: pg
                for fsdp_index, pg in self._dp_process_groups[layer.layer_id].items()
                if torch.distributed.get_rank(pg) >= 0
            }
            if process_groups:
                layer.reduce_gradients(process_groups)


class OobleckEngine:
    def __init__(
        self,
        local_rank: int,
        num_nodes: int,
        num_gpus_per_node: int,
        pipe: connection.Connection,
        args: OobleckArguments,
    ):
        assert (
            not dist.is_initialized()
        ), "torch.distributed must not be initialized when initializing OobleckEngine."

        self._agent_pipe: connection.Connection = pipe
        self._args: OobleckArguments = args
        training_args = {
            "output_dir": f"/tmp/oobleck/output/{args.model_name}-{args.model_tag}",
            "per_device_train_batch_size": args.microbatch_size,
            "no_cuda": True,  # don't use cuda in HFTrainingArguments
            "log_level": "error",  # omit warning messages from HFTrainingArguments
            # do not set gradient_accumulation_steps in HFTrainingArguments
            "max_steps": args.steps,
        }
        self._hf_training_args: HFTrainingArguments = HFTrainingArguments(
            **training_args
        )

        self._local_rank: int = local_rank
        self._num_nodes: int = num_nodes
        self._num_gpus_per_node: int = num_gpus_per_node

        # Initialize without distributed
        self._dataset: OobleckDataset
        self._model: OobleckModel
        self._pipeline_templates: list[PipelineTemplate]
        (
            self._dataset,
            self._model,
            self._profile_results,
            self._pipeline_templates,
        ) = self._initialize_engine(self._num_nodes, self._num_gpus_per_node)

    def _initialize_engine(
        self, num_nodes: int, num_gpus_per_node: int
    ) -> tuple[
        OobleckDataset,
        OobleckModel,
        LayerExecutionResults,
        list[PipelineTemplate],
    ]:
        dataset = OobleckDataset(
            self._args.model_name,
            self._args.dataset_path,
            self._args.dataset_name,
            self._args.model_args["n_positions"]
            if self._args.model_args and "n_positions" in self._args.model_args
            else None,
        )

        model = OobleckModel(
            self._args.model_name,
            dataset.sample,
            self._hf_training_args,
            self._args.model_tag,
            self._args.model_args,
        )

        profile_results: LayerExecutionResults = get_profile_results(
            self._args.model_name,
            self._args.model_tag,
            self._hf_training_args.per_device_train_batch_size,
        )

        # Minimum number of nodes is determined by the memory capacity.
        # TODO: calculate minimum number of nodes more precisely. This is still inaccurate
        total_memory_consumption = 6 * sum(
            [layer_result._mem_required[0] for layer_result in profile_results.get()]
        )
        total_memory_consumption += max(
            [layer_result._mem_required[1] for layer_result in profile_results.get()]
        )
        min_num_nodes = max(
            1,
            math.ceil(
                total_memory_consumption
                / (
                    torch.cuda.get_device_properties("cuda:0").total_memory
                    * self._num_gpus_per_node
                )
            ),
        )
        max_num_nodes = num_nodes
        assert min_num_nodes <= max_num_nodes, (
            "Minimum required number of nodes is larger than maximum number of nodes "
            f"(minimum required: {min_num_nodes}, you have: {max_num_nodes})."
        )

        logger.info(f"Number of nodes range: ({min_num_nodes}, {max_num_nodes})")

        # TODO: Calculate num_gpus_range based on profile results
        template_generator = PipelineTemplateGenerator()
        pipeline_templates: list[
            PipelineTemplate
        ] = template_generator.create_pipeline_templates(
            profile_results, (min_num_nodes, max_num_nodes), num_gpus_per_node
        )

        return dataset, model, profile_results, pipeline_templates

    def initialize_distributed(self):
        """
        At the beginning, `dist_info` is None.
        During reconfiguration, `dist_info` is given in
        `ReconfigurationEngine._on_receive_reconfiguration_notification`.
        """
        if dist.is_initialized():
            # Destroy all process groups
            # TODO: if we try to destroy a process group where some operation is stuck,
            # destroying it might be stuck as well.
            # If this is witnessed, change it to destryoing all process groups
            # manually gathered in ThreadPoolExecutor.
            logger.info("Destroying distributed process group...")
            torch.distributed.destroy_process_group(torch.distributed.group.WORLD)
            dist.cdb = None

        if not (hasattr(self, "_dist_info") and self._dist_info is not None):
            self._dist_info: DistributionInfo = self._agent_pipe.recv()
            self._rank_map: dict[str, list[int]] = {
                ip: list(
                    range(
                        i * self._num_gpus_per_node, (i + 1) * self._num_gpus_per_node
                    )
                )
                for i, ip in enumerate(self._dist_info.agent_ips)
            }
        dist_info = self._dist_info

        my_ip: str = socket.gethostbyname(socket.gethostname())
        assert my_ip in dist_info.agent_ips, "My IP is not in dist info."

        self._num_nodes = len(dist_info.agent_ips)
        self._world_size = dist_info.world_size
        self._rank = self._rank_map[my_ip][self._local_rank]

        if next(iter(self._rank_map)) == my_ip and self._local_rank == 0:
            store = torch.distributed.TCPStore(
                host_name=my_ip,
                port=0,
                world_size=dist_info.world_size,
                is_master=True,
                wait_for_workers=False,
            )
            logger.info(f"Creating a TCP store on port: {store.port}")
            self._agent_pipe.send(store.port)
            # Agent will send back this information. Discard it
            self._agent_pipe.recv()
        else:
            logger.info("Waiting for a port information...")
            # wait for rank 0's port information
            port: int = self._agent_pipe.recv()
            logger.info(f"Received torch master: {dist_info.agent_ips[0]}.{port}")
            store = torch.distributed.TCPStore(
                host_name=dist_info.agent_ips[0],
                port=port,
                world_size=dist_info.world_size,
                is_master=False,
                wait_for_workers=False,
            )
        torch.distributed.init_process_group(
            backend="nccl",
            store=store,
            rank=self._rank,
            world_size=dist_info.world_size,
        )
        dist.init_distributed(dist_backend="nccl", dist_init_required=False)
        assert torch.distributed.is_initialized() and dist.is_initialized()

        logger.info(f"[rank: {self._rank}] Distributed initialization is done.")

        # TODO: create pipeline and continue training

    def instantiate_pipelines(self, global_num_microbatch: int):
        instantiator = PipelineInstantiator()
        execution_plan: HeterogeneousPipelinesExecutionPlan = (
            instantiator.get_best_execution_plan(
                self._pipeline_templates,
                [
                    layer._allreduce_across_nodes
                    for layer in self._profile_results.get()
                ],
                self._num_nodes,
                global_num_microbatch,
            )
        )

        # TODO: get current iteration progress
        dataloader: OobleckDataLoader = OobleckDataLoader(
            args=self._hf_training_args,
            datasets=self._dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index=execution_plan.my_pipeline_index,
            num_microbatches=execution_plan.num_microbatches,
            num_iterations_done=0,
            epoch=0,
        )

        self._pipeline: OobleckPipeline
        self._pipeline, pipelines = execution_plan.instantiate(
            model=self._model,
            dataloader=dataloader,
            training_args=self._hf_training_args,
            num_gpus_per_node=self._num_gpus_per_node,
            step=0,
        )

        for pipeline in pipelines:
            pipeline.initialize_distributed_fsdp()
            pipeline.initialize_distributed_pipeline()
        self._pipeline.initialize_execution(self._model)

        assert self._pipeline.communication is not None
        assert self._pipeline.execution is not None

        self._dp_engine = DataParallelEngine(self, pipelines)
        self._reconfiguration = ReconfigurationEngine(self, pipelines)

    def _train_step(self):
        self._pipeline.train()
        self._dp_engine.do_allreduce()
        self._pipeline.execution.optimizer_step()

    def train(self):
        assert self._hf_training_args.max_steps > 0

        for step in range(self._hf_training_args.max_steps):
            logger.info(f"Step {step}")
            self._train_step()
