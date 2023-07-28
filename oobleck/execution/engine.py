from __future__ import annotations

import copy
import logging
import socket
import weakref
from collections import defaultdict
from multiprocessing import connection

import deepspeed.comm as dist
import torch.distributed
from transformers.training_args import TrainingArguments as HFTrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
    get_profile_results,
)
from oobleck.elastic.message_util import DistributionInfo
from oobleck.elastic.training_util import TrainingArguments as OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader, OobleckSampler
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)


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

    @property
    def engine(self):
        return self._engine()

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

        # Sort ranks by length so that smaller pipeline ranks always come first.
        new_ranks_list.sort(key=lambda ranks: len(ranks))

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
                            if layer._layer_id == layer_index
                        )._param_handle.flat_param
                        next(
                            layer
                            for layer in new_pipeline.execution._layers
                            if layer._layer_id == layer_index
                        )._param_handle.flat_param.data = param.data
                    else:
                        param = next(
                            layer
                            for layer in new_pipeline.execution._layers
                            if layer._layer_id == layer_index
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
        # layer_index -> dict of [fsdp_index -> list of ranks]
        dp_process_groups: dict[int, dict[int, dist.ProcessGroup]],
    ):
        self._engine = weakref.ref(engine)
        self._dp_process_groups = dp_process_groups

    @property
    def engine(self):
        return self._engine()

    def do_allreduce(self):
        for layer in self.engine._pipeline.execution._layers:
            process_groups_per_layer = self._dp_process_groups[layer._layer_id]
            layer.reduce_gradients(list(process_groups_per_layer.values()))


class OobleckEngine:
    def __init__(self, pipe: connection.Connection, args: OobleckArguments):
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
        }
        self._hf_training_args: HFTrainingArguments = HFTrainingArguments(
            **training_args
        )

        self._num_gpus_per_node: int = pipe.recv()

        # Initialize without distributed
        self._dataset: OobleckDataset
        self._model: OobleckModel
        self._pipeline_templates: list[PipelineTemplate]
        (
            self._dataset,
            self._model,
            self._profile_results,
            self._pipeline_templates,
        ) = self.initialize_engine(self._num_gpus_per_node)

    def initialize_engine(
        self, num_gpus_per_node: int
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

        num_gpus_per_node: int = torch.cuda.device_count()
        # Calculate num_gpus_range based on profile results

        template_generator = PipelineTemplateGenerator()
        pipeline_templates: list[
            PipelineTemplate
        ] = template_generator.create_pipeline_templates(
            profile_results, (1, num_gpus_per_node), num_gpus_per_node
        )

        return dataset, model, profile_results, pipeline_templates

    def initialize_distributed(self):
        if dist.is_initialized():
            # TODO: destroying process group should be done in C++ backend
            logging.info("Destroying distributed process group...")
            torch.distributed.destroy_process_group()
            dist.cdb = None

        dist_info: DistributionInfo = self._agent_pipe.recv()
        my_ip: str = socket.gethostbyname(socket.gethostname())
        assert my_ip in dist_info.agent_ips, "My IP is not in dist info."

        self._num_nodes = len(dist_info.agent_ips)
        self._world_size = dist_info.world_size
        self._local_rank = torch.cuda.current_device()
        self._rank = (
            dist_info.agent_ips.index(my_ip) * torch.cuda.device_count()
            + self._local_rank
        )

        if self._rank == 0:
            store = torch.distributed.TCPStore(
                host_name=my_ip,
                port=0,
                world_size=dist_info.world_size,
                is_master=True,
                wait_for_workers=False,
            )
            self._agent_pipe.send(store.port)
            # Agent will send back this information. Discard it
            self._agent_pipe.recv()
        else:
            # wait for rank 0's port information
            port: int = self._agent_pipe.recv()
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
        # layer_index -> dict of [fsdp_index -> list of ranks]
        process_groups_dp: dict[int, dict[int, dist.ProcessGroup]]
        self._pipeline, pipelines, process_groups_dp = execution_plan.instantiate(
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

        self._dp_engine = DataParallelEngine(self, process_groups_dp)
        self._reconfiguration = ReconfigurationEngine(self, pipelines)

    def _train_step(self):
        self._pipeline.train()
        self._dp_engine.do_allreduce()
        self._pipeline.execution.optimizer_step()

    def train(self):
        assert self._hf_training_args.max_steps > 0

        for _ in range(self._hf_training_args.max_steps):
            self._train_step()
