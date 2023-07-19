from __future__ import annotations

import logging
import socket
import weakref
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
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)


class DataParallelEngine:
    def __init__(
        self,
        engine: OobleckEngine,
        dp_process_groups: dict[int, dict[int, dist.ProcessGroup]],
    ):
        self._engine = weakref.ref(engine)
        self._dp_process_groups = dp_process_groups

    @property
    def engine(self):
        return self._engine()

    def do_allreduce(self):
        for layer, pgs_per_layer in zip(
            self.engine._pipeline.execution._layers,
            self._dp_process_groups.values(),
        ):
            layer.reduce_gradients(pgs_per_layer.values())


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

        self._max_num_gpus: int = pipe.recv()

        # Initialize without distributed
        self._dataset: OobleckDataset
        self._model: OobleckModel
        self._pipeline_templates: list[PipelineTemplate]
        (
            self._dataset,
            self._model,
            self._profile_results,
            self._pipeline_templates,
        ) = self.initialize_engine(self._max_num_gpus)

    def initialize_engine(
        self, max_num_gpus: int
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
            profile_results, (1, max_num_gpus), num_gpus_per_node
        )

        return dataset, model, profile_results, pipeline_templates

    def initialize_distributed(self):
        if dist.is_initialized():
            # TODO: destroying process group should be done in C++ backend
            logging.info("Destroying distributed process group...")
            torch.distributed.destroy_process_group()
            dist.destroy_process_group()
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
        self._dataloader: OobleckDataLoader = OobleckDataLoader(
            args=self._hf_training_args,
            datasets=self._dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index=execution_plan.my_pipeline_index,
            num_microbatches=execution_plan.num_microbatches,
            num_iterations_done=0,
            epoch=0,
        )

        self._pipeline: OobleckPipeline
        # (layer_index, fsdp_index) -> list of ranks
        process_groups_dp: dict[tuple[int, int], dist.ProcessGroup]
        self._pipeline, process_groups_dp = execution_plan.instantiate(
            model=self._model,
            dataloader=self._dataloader,
            training_args=self._hf_training_args,
            step=0,
        )

        self._dp_engine = DataParallelEngine(self, process_groups_dp)
