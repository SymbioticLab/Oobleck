import copy
import gc
import itertools
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    get_param_info,
)
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.shardformer import ShardConfig
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.plugin.heterogeneous_dataloader import HeterogeneousDataLoader
from oobleck_colossalai.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from oobleck_colossalai.process_group_mesh import PP_AXIS, HeterogeneousProcessGroupMesh
from oobleck_colossalai.shardformer.shard.placeholder import ParameterPlaceholder
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator


class OobleckPlugin(HeterogeneousParallelPlugin):
    """Plugin for Oobleck, an extension of heterogeneous parallel plugin
    to support fault tolerance and reconfiguration.
    """

    def __init__(
        self,
        tp_size: int,
        global_batch_size: int,
        microbatch_size: int,
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_used: bool = False,
    ):
        super().__init__(
            tp_size=tp_size,
            global_batch_size=global_batch_size,
            microbatch_size=microbatch_size,
            precision=precision,
            zero_stage=0,
            enable_all_optimization=False,
            enable_fused_normalization=enable_fused_normalization,
            enable_flash_attention=enable_flash_attention,
            enable_jit_fused=enable_jit_used,
            cpu_offload=False,
            communication_dtype=None,
            overlap_communication=False,
        )

    def on_receive_reconfiguration_notification(self):
        """
        A failure event is received from any worker.
        The reconfiguration engine should reconfigure affected pipelines
        using the set of pipeline templates.
        This function is called in such a case.
        """
        pass

    @torch.no_grad()
    def reconfigure(
        self,
        pipeline_templates: dict[int, PipelineTemplate],
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: HeterogeneousDataLoader,
        lr_scheduler: LRScheduler | None = None,
    ) -> tuple[
        ModelWrapper,
        OptimizerWrapper,
        DataLoader,
        LRScheduler,
    ]:
        if not hasattr(self, "pg_mesh"):
            logger.warning(
                "Received reconfiguration notification before plugin configuration."
            )
            return

        configuration_engine = ConfigurationEngine.get_instance()

        # Get old attributes
        old_pg_mesh = copy.deepcopy(self.pg_mesh.mesh)
        old_rank_map = copy.deepcopy(configuration_engine.rank_map)
        old_pipelines = copy.deepcopy(self.pipelines)

        num_layers = self.pipelines[0].num_layers
        old_my_layers = torch.tensor(
            [
                True
                if index in [coord[PP_AXIS] for coord in self.pg_mesh.coords]
                else False
                for index in range(num_layers)
            ],
            dtype=torch.bool,
            device="cuda",
        )

        # Reset process group
        configuration_engine.get_host_update()
        del self.pg_mesh
        self.pg_mesh = None
        gc.collect()
        configuration_engine.init_distributed()

        # Create a rank conversion map (old rank -> new rank)
        # TODO: create a dedicated function for this
        rank_conversion_map = {}
        for host_info, ranks in old_rank_map.items():
            if host_info not in configuration_engine.rank_map:
                continue
            for old_rank, new_rank in zip(
                ranks, configuration_engine.rank_map[host_info]
            ):
                rank_conversion_map[old_rank] = new_rank

        # each tensor indicates which layers are held by each (new) rank
        old_layers_per_rank = torch.zeros(
            configuration_engine.world_size, num_layers, dtype=torch.bool, device="cuda"
        )
        dist.all_gather_into_tensor(old_layers_per_rank, old_my_layers)
        del old_my_layers

        # Prepare layer transfer before setting new pipelines
        # Setting new pipelines will free unused layers, which some other may need.

        # Create new pipelines.
        # TODO: If there is no feasible pipeline, merge pipelines
        num_hosts_per_pipeline = [pipeline.num_stages for pipeline in old_pipelines]
        removed_ranks_list = [
            old_rank_map[host]
            for host in old_rank_map
            if host not in configuration_engine.rank_map
        ]
        for removed_ranks in removed_ranks_list:
            for pipeline_index, ranks_in_mesh in enumerate(old_pg_mesh):
                if removed_ranks in ranks_in_mesh:
                    num_hosts_per_pipeline[pipeline_index] -= 1

        new_pipelines = [
            pipeline_templates[num_stages] for num_stages in num_hosts_per_pipeline
        ]

        new_pg_mesh = HeterogeneousProcessGroupMesh(
            new_pipelines, self.shard_config.tensor_parallel_size
        )

        new_my_layers = torch.tensor(
            [
                True
                if index in [coord[PP_AXIS] for coord in new_pg_mesh.coords]
                else False
                for index in range(num_layers)
            ],
            dtype=torch.bool,
            device="cuda",
        )

        new_layers_per_rank = torch.zeros(
            configuration_engine.world_size, num_layers, dtype=torch.bool, device="cuda"
        )
        dist.all_gather_into_tensor(new_layers_per_rank, new_my_layers)
        del new_my_layers

        # Calculate required layers (layers that are now need but not held)
        # layer_index -> list of ranks
        layers_required_by_rank: defaultdict[int, list[int]] = defaultdict(list)
        for index, has_layer in np.ndenumerate(np.array(old_layers_per_rank.cpu())):
            if not has_layer and new_layers_per_rank[index]:
                rank, layer_index = index
                layers_required_by_rank[layer_index].append(rank)

        # Implement copy plan
        send_recv_ops: list[dist.P2POp] = []

        layers = list(
            itertools.chain.from_iterable(self.pipelines[0].modules_per_stage)
        )
        layer_modules: dict[str, nn.Module] = {
            layer_name: module
            for name, module in model.module.named_modules()
            for layer_name in layers
            if name == layer_name
        }

        def get_layer_holders() -> list[int]:
            layer_holders = []
            for rank, has_layer in enumerate(old_layers_per_rank[:, layer_index]):
                if has_layer:
                    layer_holders.append(rank)
            return layer_holders

        # TODO: copy model parameters from optimizer, since they are master parameters
        # that can be downcasted and used for model parameters.
        # Attributes to update
        # 1. model's parameters (placeholder -> torch.nn.Parameter with data)
        # 2. optimizer's param_info
        #    - param2id
        #    - id2param
        #    - param2shape
        # 3. optimizer.optim's state (optimizer states for each parameter)
        # 4. optimizer.optim's param_groups (master parameters in FP32)
        # 5. if optimizer is an instance of MixedPrecisionOptimizer,
        #    then optimizer's master_to_working_map and working_to_master_map
        # 6. MixedPrecisionMixin has self.params (working params)
        #    which MixedPrecisionOptimizer has as self.mixed_precision
        for layer_index, ranks in layers_required_by_rank.items():
            layer_holders = get_layer_holders()
            if not layer_holders:
                logger.error(
                    f"Currently no rank has layer: {layers[layer_index]}. "
                    "Please rerun training from last checkpoint."
                )
                raise RuntimeError(f"No one holds the layer: {layers[layer_index]}!")

            for rank_need_layer in ranks:
                if not layer_holders:
                    # Empty holder "here" means that all holders is consumed and has a job.
                    # Add more jobs to each holder.
                    layer_holders = get_layer_holders()

                sender_rank = layer_holders.pop(0)
                if configuration_engine.rank == rank_need_layer:
                    module = layer_modules[layers[layer_index]]
                    parameters: list[tuple[str, ParameterPlaceholder]] = [
                        (name, param)
                        for name, param in module.named_parameters()
                        if isinstance(param, ParameterPlaceholder)
                    ]
                    for name, param in parameters:
                        # Create new parameter and replace the placeholder
                        new_param = param.create()
                        setattr(module, name, new_param)

                        # Receive the parameter from the sender
                        send_recv_ops.append(
                            dist.P2POp(dist.irecv, new_param.data, sender_rank)
                        )

                    pass
                elif sender_rank == configuration_engine.rank:
                    module = layer_modules[layers[layer_index]]
                    for name, param in module.named_parameters():
                        send_recv_ops.append(
                            dist.P2POp(dist.isend, param.data, rank_need_layer)
                        )

        works = dist.batch_isend_irecv(send_recv_ops)
        for work in works:
            work.wait()

        # free no-longer necessary parameter here

        # Redistribute microbatches
        instantiator = PipelineInstantiator(new_pipelines, self.global_batch_size)
        _, num_microbatches = instantiator.distribute_batch(
            dict(Counter(new_pipelines))
        )

        if isinstance(self.shard_config, ShardConfig):
            self.shard_config = dict(
                tp_size=self.tp_size,
                enable_all_optimization=self.shard_config.enable_all_optimization,
                enable_fused_normalization=self.shard_config.enable_fused_normalization,
                enable_flash_attention=self.shard_config.enable_flash_attention,
                enable_jit_fused=self.shard_config.enable_jit_fused,
                enable_sequence_parallelism=False,
                enable_sequence_overlap=False,
            )
        self.set_pipelines(new_pipelines, num_microbatches)

        # recreate optimizer, dataloader, etc

        # TODO: this re-initializing optimizer may end up losing internal states.
        # Must copy those optimizer states as well.
        param_info = get_param_info(optimizer.optim)
        if self.precision in ["fp16", "bf16"]:
            optimizer = HybridParallelAMPOptimizer(
                optimizer.optim,
                model,
                use_pipeline=self.enable_pipeline_parallelism,
                param_info=param_info,
                precision=self.precision,
                max_norm=self.max_norm,
                pp_process_group=self.pp_group,
                tp_process_group=self.tp_group,
                **self.amp_config,
            )
        else:
            optimizer = HybridParallelNaiveOptimizer(
                optimizer.optim,
                model,
                use_pipeline=self.enable_pipeline_parallelism,
                param_info=param_info,
                max_norm=self.max_norm,
                pp_process_group=self.pp_group,
                tp_process_group=self.tp_group,
            )
        num_microbatches = [
            self.num_microbatches[pipeline] for pipeline in self.pipelines
        ]

        return model, optimizer, dataloader, lr_scheduler
