import copy
import gc
import io
import itertools
from collections import Counter, defaultdict
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from loguru import logger
from oobleck_colossalai import (
    HeterogeneousDataLoader,
    HeterogeneousParallelModule,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)
from oobleck_colossalai.process_group_mesh import (
    DP_AXIS,
    PP_AXIS,
    TP_AXIS,
    HeterogeneousProcessGroupMesh,
)
from oobleck_colossalai.shardformer.shard.placeholder import TensorPlaceholder
from oobleck_colossalai.shardformer.shard.shardformer import ModelSharder
from oobleck_colossalai.stage_manager import HeterogeneousPipelineStageManager
from torch.optim.lr_scheduler import LRScheduler

from oobleck.engine.configuration_engine import ConfigurationEngine, HostInfo
from oobleck.engine.pipeline_instantiator import PipelineInstantiator


class OobleckPlugin(HeterogeneousParallelPlugin):
    """Plugin for Oobleck, an extension of heterogeneous parallel plugin
    to support fault tolerance and reconfiguration.
    """

    def __init__(
        self,
        pipeline_templates: list[PipelineTemplate],
        tp_size: int,
        global_batch_size: int,
        microbatch_size: int,
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_used: bool = False,
    ):
        assert (
            global_batch_size % microbatch_size == 0
        ), "Global batch size must be divisible by microbatch size. "
        pipelines, num_microbatches = self._instantiate_pipelines(
            pipeline_templates=pipeline_templates,
            global_num_microbatches=global_batch_size // microbatch_size,
        )

        super().__init__(
            pipelines=pipelines,
            tp_size=tp_size,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
            precision=precision,
            enable_all_optimization=False,
            enable_fused_normalization=enable_fused_normalization,
            enable_flash_attention=enable_flash_attention,
            enable_jit_fused=enable_jit_used,
        )

        self.pipeline_templates = pipeline_templates

    def on_receive_reconfiguration_notification(self):
        """
        A failure event is received from any worker.
        The reconfiguration engine should reconfigure affected pipelines
        using the set of pipeline templates.
        This function is called in such a case.
        """
        pass

    def _instantiate_pipelines(
        self,
        pipeline_templates: list[PipelineTemplate],
        global_num_microbatches: int,
        old_pg_mesh: Optional[list] = None,
        old_rank_map: Optional[dict[HostInfo, list[int]]] = None,
    ) -> tuple[list[PipelineTemplate], dict[PipelineTemplate, int]]:
        """
        Get pipelines to be instantiated from the list of pipeline templates.
        Each pipeline template may be instantiated several times.

        Args:
            pipeline_templates: List of pipeline templates
            global_num_microbatches: Number of microbatches in a global batch
            old_pg_mesh: previous group rank mesh (HeterogeneousProcessGroupMesh.mesh).
                Necessary for reconfiguration. None if not reconfiguration.
            old_rank_map: previous rank map (ConfigurationEngine.rank_map).
                Necessary for reconfiguration. None if not reconfiguration.

        Returns:
            list[PipelineTemplate]: List of pipelines to be instantiated
            dict[PipelineTemplate, int]: Number of microbatches for each pipeline
        """
        pipeline_instantiator = PipelineInstantiator(
            pipeline_templates, global_num_microbatches
        )
        configuration_engine = ConfigurationEngine.get_instance()

        # Is this for reconfiguration?
        if old_pg_mesh is not None and old_rank_map is not None:
            if not hasattr(self, "pipelines"):
                raise RuntimeError(
                    "Existing pipelines are necessary for reconfiguration"
                )

            num_hosts_per_pipeline = [
                pipeline.num_stages for pipeline in self.pipelines
            ]
            removed_ranks_list = [
                old_rank_map[host]
                for host in old_rank_map
                if host not in configuration_engine.rank_map
            ]

            for removed_ranks in removed_ranks_list:
                for pipeline_index, ranks_in_mesh in enumerate(old_pg_mesh):
                    if removed_ranks in ranks_in_mesh:
                        num_hosts_per_pipeline[pipeline_index] -= 1

            # TODO (insujang): If there is no feasible pipeline, merge pipelines
            pipelines = [
                pipeline_templates[num_stages] for num_stages in num_hosts_per_pipeline
            ]

            _, num_microbatches = pipeline_instantiator.distribute_batch(
                dict(Counter(pipelines))
            )
        else:
            num_instances, num_microbatches = pipeline_instantiator.instantiate(
                len(configuration_engine.dist_info)
            )

            pipelines = list(
                itertools.chain.from_iterable(
                    itertools.repeat(template, num_templates)
                    for template, num_templates in num_instances.items()
                )
            )

        return pipelines, num_microbatches

    @torch.no_grad()
    def reconfigure(
        self,
        pipeline_templates: dict[int, PipelineTemplate],
        model: HeterogeneousParallelModule,
        optimizer: HybridParallelAMPOptimizer | HybridParallelNaiveOptimizer,
        dataloader: HeterogeneousDataLoader,
        lr_scheduler: LRScheduler | None = None,
    ):
        configuration_engine = ConfigurationEngine.get_instance()
        device = get_accelerator().get_current_device()

        # all layers in the model
        layers = list(
            itertools.chain.from_iterable(self.pipelines[0].modules_per_stage)
        )
        num_layers = len(layers)

        # Backup old attributes for reconfiguration before resetting
        old_pg_mesh = copy.deepcopy(self.pg_mesh.mesh)
        old_rank_map = copy.deepcopy(configuration_engine.rank_map)

        old_my_layers = torch.tensor(
            [
                True
                if index in [coord[PP_AXIS] for coord in self.pg_mesh.coords]
                else False
                for index in range(num_layers)
            ],
            dtype=torch.bool,
            device=device,
        )

        # Reset process group
        configuration_engine.get_host_update()
        del self.pg_mesh
        self.pg_mesh = None
        gc.collect()
        configuration_engine.init_distributed()

        # each tensor indicates which layers are held by each (new) rank
        old_layers_per_rank = torch.zeros(
            configuration_engine.world_size, num_layers, dtype=torch.bool, device=device
        )
        dist.all_gather_into_tensor(old_layers_per_rank, old_my_layers)
        del old_my_layers

        # Prepare layer transfer before setting new pipelines
        # Setting new pipelines will free unused layers, which some other may need.

        # Create new pipelines.
        # TODO: If there is no feasible pipeline, merge pipelines
        new_pipelines, new_num_microbatches = self._instantiate_pipelines(
            pipeline_templates,
            self.global_batch_size // self.microbatch_size,
            old_pg_mesh,
            old_rank_map,
        )

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
            device=device,
        )

        new_layers_per_rank = torch.zeros(
            configuration_engine.world_size, num_layers, dtype=torch.bool, device=device
        )
        dist.all_gather_into_tensor(new_layers_per_rank, new_my_layers)
        del new_my_layers

        """
        Missing layer transfer based on new pipeline configuration
        old_layers_per_rank (torch.Tensor[world_size, num_layers]):
            For each rank, indicates which layers are held by the rank.
            True if the rank holds the layer, False otherwise.
            world_size is the size after failures, and ranks are newly assigned ones.
        new_layers_per_rank (torch.Tensor[world_size, num_layers]):
            For each rank, indicates which layers should be held by the rank.
            True if the rank should hold the layer, False if should not.

        1. Calculate required layers (layers that are now need but not held)
        2. For each required layer, find a rank that holds the layer
        3. Send the layer to the rank that needs the layer
        4. Update optimizer states and model parameters
        5. Calculate non-required layers (layers that are held but not needed)
        6. Free non-required layers
        """

        # Calculate required layers (layers that are now need but not held)
        # layer_index -> list of ranks
        layers_required_by_rank: dict[int, list[int]] = defaultdict(list)
        for index, has_layer in np.ndenumerate(old_layers_per_rank.numpy(force=True)):
            if not has_layer and new_layers_per_rank[index]:
                rank, layer_index = index
                layers_required_by_rank[layer_index].append(rank)

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

                    buffer_placeholders: dict[str, TensorPlaceholder] = getattr(
                        module, "_buffer_placeholders"
                    )
                    for name, buffer_holder in buffer_placeholders.items():
                        buffer_holder: TensorPlaceholder = cast(
                            TensorPlaceholder, buffer_holder
                        )
                        setattr(module, name, buffer_holder.create())
                    delattr(module, "_buffer_placeholders")

                    param_placeholders: dict[str, TensorPlaceholder] = getattr(
                        module, "_parameter_placeholders"
                    )
                    for name, param_holder in param_placeholders.items():
                        param_holder: TensorPlaceholder = cast(
                            TensorPlaceholder, param_holder
                        )

                        # Check: whether shape is correct
                        param_info: dict[str, Any] = optimizer.param_info
                        assert (
                            param_info["param2shape"][param_holder.param_id]
                            == param_holder.shape
                        )

                        # Get size of optimizer state
                        state_size: torch.Tensor = torch.empty(
                            1, dtype=torch.int64, device=device
                        )
                        dist.recv(state_size, sender_rank)

                        # Get pickled optimizer state for parameter
                        state_tensor: torch.Tensor = torch.empty(
                            state_size.item(), dtype=torch.uint8, device=device
                        )
                        dist.recv(state_tensor, sender_rank)
                        buff = io.BytesIO(state_tensor.cpu().numpy())
                        state_tensor: torch.Tensor = torch.load(buff)

                        param_tensor = param_holder.create(dtype=torch.float32)
                        dist.recv(param_tensor, sender_rank)

                        # Set the model parameter
                        model_param = torch.nn.Parameter(
                            param_tensor.to(dtype=model.mixed_precision)
                            if self.precision in ["fp16", "bf16"]
                            else param_tensor
                        )
                        setattr(module, name, model_param)

                        # Set the ColossalAI Optimizer param_info
                        optimizer.optim.state[param_tensor] = state_tensor
                        optimizer.optim.param_groups[0]["params"].append(param_tensor)

                        optim_param_index = param_info["param2id"][
                            param_holder.param_id
                        ]
                        del optimizer.param_info["param2id"][param_holder.param_id]
                        optimizer.param_info["param2id"][id(param_tensor)] = (
                            optim_param_index
                        )
                        optimizer.param_info["id2param"][optim_param_index] = id(
                            param_tensor
                        )
                        optimizer.master_to_working_map[param_tensor] = model_param
                        optimizer.working_to_master_map[model_param] = param_tensor
                    delattr(module, "_parameter_placeholders")

                elif sender_rank == configuration_engine.rank:
                    module = layer_modules[layers[layer_index]]
                    for name, param in module.named_parameters(recurse=False):
                        # Find master weights
                        master_param = (
                            optimizer.working_to_master_map[param]
                            if isinstance(optimizer, HybridParallelAMPOptimizer)
                            else param
                        )

                        # Pickle optimizer state
                        buff = io.BytesIO()
                        torch.save(optimizer.optim.state[master_param], buff)
                        buff.seek(0)

                        state_tensor = torch.frombuffer(
                            buff.getbuffer(), dtype=torch.uint8
                        ).to(device)
                        state_size = torch.tensor(
                            [state_tensor.numel()], dtype=torch.int64, device=device
                        )

                        dist.send(state_size, rank_need_layer)
                        dist.send(state_tensor, rank_need_layer)

                        # Send the parameter
                        dist.send(master_param, rank_need_layer)

        # free no-longer necessary parameter here
        # TODO: must create _buffer_placeholders and _parameter_placeholders
        old_layers = old_layers_per_rank.numpy(force=True)[:, configuration_engine.rank]
        new_layers = new_layers_per_rank.numpy(force=True)[:, configuration_engine.rank]
        for index, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
            if old_layer and not new_layer:
                module = layer_modules[layers[index]]
                ModelSharder.set_tensors_to_placeholder(module)

        # Update plugin attributes
        self.pipelines = new_pipelines
        self.num_microbatches = new_num_microbatches
        self.pg_mesh = new_pg_mesh
        self.stage_manager = HeterogeneousPipelineStageManager(self.pg_mesh, PP_AXIS)
        self.dp_groups = self.pg_mesh.get_group_along_axis(DP_AXIS)
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS)
        self.pp_group = self.pg_mesh.get_group_along_axis(PP_AXIS)

        self._pipeline_index = self.pg_mesh.coords[0][DP_AXIS]
        self.schedule = OneForwardOneBackwardSchedule(
            stage_manager=self.stage_manager,
            microbatch_size=self.microbatch_size,
        )

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager

        # TODO: how to reset shared parameters? :/
        model.stage_manager = self.stage_manager
        my_pipeline = self.pipelines[self._pipeline_index]
        module_names = my_pipeline.modules_per_stage[self.stage_manager.stage]
        model.dp_groups = (
            {
                module_name: dp_group
                for module_name, dp_group in zip(module_names, self.dp_groups)
            }
            if self.dp_groups
            else None
        )
        model.tp_group = self.tp_group

        from oobleck_colossalai.shardformer.policies.auto_policy import get_autopolicy
        from oobleck_colossalai.shardformer.shard.shardformer import ShardFormer

        policy = get_autopolicy(my_pipeline.model_name)
        policy.set_model(model.module)
        policy.set_pipeline_template(my_pipeline)
        policy.set_shard_config(self.shard_config)
        shardformer = ShardFormer(self.shard_config)
        model.module, model.shared_params = shardformer.optimize(model.module, policy)
        pass
