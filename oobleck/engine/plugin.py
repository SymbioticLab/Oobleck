import copy
import gc
import io
import itertools
from collections import Counter, defaultdict
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
)
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.plugin.heterogeneous_dataloader import HeterogeneousDataLoader
from oobleck_colossalai.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from oobleck_colossalai.process_group_mesh import (
    DP_AXIS,
    PP_AXIS,
    TP_AXIS,
    HeterogeneousProcessGroupMesh,
)
from oobleck_colossalai.shardformer.shard.placeholder import TensorPlaceholder
from oobleck_colossalai.stage_manager import HeterogeneousPipelineStageManager
from torch.optim.lr_scheduler import LRScheduler

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator


class OobleckPlugin(HeterogeneousParallelPlugin):
    """Plugin for Oobleck, an extension of heterogeneous parallel plugin
    to support fault tolerance and reconfiguration.
    """

    def __init__(
        self,
        pipelines: list[PipelineTemplate],
        tp_size: int,
        microbatch_size: int,
        num_microbatches: dict[PipelineTemplate, int],
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_used: bool = False,
    ):
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
    ):
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
                device = get_accelerator().get_current_device()
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

                        # Get state size
                        state_size: torch.Tensor = torch.empty(
                            1, dtype=torch.int64, device=device
                        )
                        dist.recv(state_size, sender_rank)
                        # Get pickled state tensor
                        state_tensor: torch.Tensor = torch.empty(
                            state_size.item(), dtype=torch.uint8, device=device
                        )
                        dist.recv(state_tensor, sender_rank)
                        buff = io.BytesIO(state_tensor.cpu().numpy())
                        state_tensor: torch.Tensor = torch.load(buff)

                        param_tensor = param_holder.create(dtype=torch.float32)
                        dist.recv(param_tensor, sender_rank)

                        model_param = torch.nn.Parameter(
                            param_tensor.to(param_holder.precision)
                            if param_holder.precision == torch.float32
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
                        optimizer.master_to_working_map[id(param_tensor)] = model_param
                        optimizer.working_to_master_map[model_param] = id(param_tensor)
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

        # Redistribute microbatches
        instantiator = PipelineInstantiator(new_pipelines, self.global_batch_size)
        _, num_microbatches = instantiator.distribute_batch(
            dict(Counter(new_pipelines))
        )

        self.pipelines = new_pipelines
        self.num_microbatches = num_microbatches
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
