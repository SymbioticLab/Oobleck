import logging
import os
import socket
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


class OobleckEngine:
    def __init__(self, pipe: connection.Connection, args: OobleckArguments):
        assert (
            not dist.is_initialized()
        ), "torch.distributed must not be initialized when initializing OobleckEngine."

        self._agent_pipe: connection.Connection = pipe
        self._args: OobleckArguments = args

        if args.hf_training_args is None:
            args.hf_training_args = {}
        args.hf_training_args["output_dir"] = "/tmp/oobleck_output"

        self._hf_training_args: HFTrainingArguments = HFTrainingArguments(
            **args.hf_training_args
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
        self._local_rank = int(os.environ["CUDA_VISIBLE_DEVICES"])
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
                host_name=my_ip,
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

    def instantiate_pipelines(self):
        instantiator = PipelineInstantiator()
        execution_plan: HeterogeneousPipelinesExecutionPlan = (
            instantiator.get_best_execution_plan(
                self._pipeline_templates,
                [
                    layer._allreduce_across_nodes
                    for layer in self._profile_results.get()
                ],
                self._num_nodes,
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

        self._pipeline: OobleckPipeline = execution_plan.instantiate(
            model=self._model,
            dataloader=self._dataloader,
            training_args=self._hf_training_args,
            step=0,
        )
