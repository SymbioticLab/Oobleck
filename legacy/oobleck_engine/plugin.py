import copy
import io
import itertools
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    TP_AXIS,
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from colossalai.shardformer.layer.parallel_module import ParallelModule
from cornstarch import (
    HeterogeneousDataLoader,
    HeterogeneousParallelModule,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)
from cornstarch.process_group_mesh import (
    PP_AXIS,
    HeterogeneousProcessGroupMesh,
)
from cornstarch.shardformer.shard.shardformer import ModelSharder
from loguru import logger
from torch.optim.lr_scheduler import LRScheduler

from oobleck.engine.configuration_engine import ConfigurationEngine, HostInfo
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
        fault_tolerance_threshold: int = 3,
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_used: bool = False,
    ):
        assert (
            global_batch_size % microbatch_size == 0
        ), "Global batch size must be divisible by microbatch size. "

        configuration_engine = ConfigurationEngine.get_instance()
        assert all(
            len(host.devices.split(",")) == tp_size
            for host in configuration_engine.dist_info
        ), f"All agent must have the same worker size {tp_size}."

        super().__init__(
            tp_size=tp_size,
            microbatch_size=microbatch_size,
            precision=precision,
            enable_fused_normalization=enable_fused_normalization,
            enable_flash_attention=enable_flash_attention,
            enable_jit_fused=enable_jit_used,
        )

        self.global_batch_size = global_batch_size
        self.fault_tolerance_threshold = fault_tolerance_threshold

    def _instantiate_pipelines(
        self,
        pipeline_templates: dict[int, PipelineTemplate],
        global_num_microbatches: int,
        old_pg_mesh: Optional[list] = None,
        old_rank_map: Optional[dict[HostInfo, list[int]]] = None,
    ) -> tuple[list[PipelineTemplate], dict[PipelineTemplate, int]]:
        """
        Get pipelines to be instantiated from the list of pipeline templates.
        Each pipeline template may be instantiated several times.

        Args:
            pipeline_templates: Dict of pipeline templates (num_hosts -> pipeline template)
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
            pipeline_templates, global_num_microbatches, self.fault_tolerance_threshold
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
                pipeline_templates[num_stages]
                for num_stages in num_hosts_per_pipeline
                if num_stages > 0
            ]

            _, num_microbatches = pipeline_instantiator.distribute_batch(
                dict(Counter(pipelines)), need_all_pipelines_have_batch=True
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
    ) -> tuple[
        HeterogeneousParallelPlugin,
        HybridParallelAMPOptimizer | HybridParallelNaiveOptimizer,
        HeterogeneousDataLoader,
        LRScheduler | None,
    ]:
        configuration_engine = ConfigurationEngine.get_instance()
        device = get_accelerator().get_current_device()

        # all layers in the model
        layers = list(
            itertools.chain.from_iterable(self.pipelines[0].modules_per_stage)
        )
        num_layers = len(layers)

        # Backup old attributes for reconfiguration before resetting
        old_pg_mesh: np.ndarray = copy.deepcopy(self.pg_mesh.mesh)
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
        configuration_engine.init_distributed()

        def all_gather_layers_per_rank(
            data: torch.Tensor, device: torch.device
        ) -> dict[tuple[int], np.ndarray]:
            data_list = torch.empty(
                configuration_engine.world_size,
                *data.shape,
                device=device,
                dtype=data.dtype,
            )
            dist.all_gather_into_tensor(data_list, data)

            data_list = data_list.numpy(force=True)
            result = {}
            for i in range(
                0,
                configuration_engine.world_size,
                self.shard_config.tensor_parallel_size,
            ):
                ranks = list(range(i, i + self.shard_config.tensor_parallel_size))
                assert all(
                    np.array_equal(data_list[i], data_list[i + tp_index])
                    for tp_index in range(self.shard_config.tensor_parallel_size)
                ), f"Data mismatch between ranks in the same tensor parallel group: {ranks}."
                result[tuple(ranks)] = data_list[i]

            return result

        # each tensor indicates which layers are held by each (new) rank
        old_layers_per_ranks = all_gather_layers_per_rank(old_my_layers, device)
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

        new_layers_per_ranks = all_gather_layers_per_rank(new_my_layers, device)

        """
        Missing layer transfer based on new pipeline configuration
        old_layers_per_dp_rank (numpy.ndarray[dp_size, num_layers]):
            For each dp rank, indicates which layers are held by the replica.
            True if the dp group holds the layer, False otherwise.
            dp_size is the size after failures which is newly assigned.
        new_layers_per_dp_rank (numpy.ndarray[dp_size, num_layers]):
            For each dp rank, indicates which layers should be held by the replica.
            True if the dp group should hold the layer, False if should not.

        1. Calculate required layers (layers that are now need but not held)
        2. For each required layer, find a rank that holds the layer
        3. Send the layer to the rank that needs the layer
        4. Update optimizer states and model parameters
        5. Calculate non-required layers (layers that are held but not needed)
        6. Free non-required layers
        """

        # Calculate required layers (layers that are now need but not held)
        # layer_index -> list of tuples of ranks (each tuple includes ranks per tensor parallel group)
        # Even if tensor parallelism is not used, it is a tuple with one element.
        layers_required_by_ranks: dict[int, list[tuple[int]]] = defaultdict(list)
        for ranks, has_layers in old_layers_per_ranks.items():
            for layer_index, has_layer in enumerate(has_layers):
                if not has_layer and new_layers_per_ranks[ranks][layer_index]:
                    layers_required_by_ranks[layer_index].append(ranks)

        layer_modules: dict[str, nn.Module] = {
            layer_name: module
            for name, module in model.module.named_modules()
            for layer_name in layers
            if name == layer_name
        }

        def get_layer_holder_ranks(layer_index: int) -> list[tuple[int]]:
            layer_holder_ranks = []
            for ranks, has_layers in old_layers_per_ranks.items():
                if has_layers[layer_index]:
                    layer_holder_ranks.append(ranks)
            return layer_holder_ranks

        adds_to_param_info: dict[str, dict] = {
            "param2id": {},
            "id2param": {},
            "param2shape": {},
        }
        removes_from_param_info: dict[str, list] = {
            "param2id": [],
            "id2param": [],
            "param2shape": [],
        }

        for layer_index, ranks in layers_required_by_ranks.items():
            layer_holder_ranks = get_layer_holder_ranks(layer_index)
            if not layer_holder_ranks:
                logger.error(
                    f"Currently no rank has layer: {layers[layer_index]}. "
                    "Please rerun training from last checkpoint."
                )
                raise RuntimeError(f"No one holds the layer: {layers[layer_index]}!")

            module = layer_modules[layers[layer_index]]

            for ranks_need_layer in ranks:
                if not layer_holder_ranks:
                    # Empty holder "here" means that all holders is consumed and has a job.
                    # Add more jobs to each holder.
                    layer_holder_ranks = get_layer_holder_ranks(layer_index)

                sender_ranks = layer_holder_ranks.pop(0)
                if configuration_engine.rank in ranks_need_layer:
                    # Find the corresponding tensor parallel rank in sender_ranks
                    sender_rank = sender_ranks[
                        ranks_need_layer.index(configuration_engine.rank)
                    ]
                    for (
                        submodule,
                        name,
                        placeholder,
                    ) in ModelSharder.buffer_placeholders(
                        module, delete_placeholders_after=True
                    ):
                        setattr(submodule, name, placeholder.create())

                    for (
                        submodule,
                        name,
                        placeholder,
                    ) in ModelSharder.parameter_placeholders(
                        module, delete_placeholders_after=True
                    ):
                        size: torch.Tensor = torch.empty(
                            1, dtype=torch.int64, device=device
                        )
                        dist.recv(size, sender_rank)

                        tensor: torch.Tensor = torch.empty(
                            size.item(), dtype=torch.uint8, device=device
                        )
                        dist.recv(tensor, sender_rank)
                        buff = io.BytesIO(tensor.numpy(force=True))
                        tensor: dict[str, tuple[dict, torch.Tensor]] = torch.load(
                            buff, map_location=device
                        )

                        states: dict[str, torch.Tensor] = tensor["states"]
                        master_tensor: torch.Tensor = tensor["parameter"]

                        # Check whether shape is correct
                        assert master_tensor.shape == placeholder.shape, (
                            f"{name} Shape mismatch between placeholder and parameter. "
                            f"received parameter shape: {master_tensor.shape}, "
                            f"placeholder shape: {placeholder.shape}"
                        )

                        p = torch.nn.Parameter(
                            master_tensor.to(dtype=model.mixed_precision)
                            if isinstance(optimizer, MixedPrecisionOptimizer)
                            and master_tensor.dtype == torch.float32
                            else master_tensor
                        )
                        setattr(submodule, name, p)

                        # TODO: check if it is still tensor (not nn.parameter) if we do fp32 training
                        optimizer.optim.state[master_tensor] = states
                        optimizer.optim.param_groups[0]["params"].append(master_tensor)

                        if isinstance(optimizer, MixedPrecisionOptimizer):
                            optimizer.master_to_working_map[master_tensor] = p
                            optimizer.working_to_master_map[p] = master_tensor

                        # Cache updates and apply them later
                        # This is because of potential duplicated IDs:
                        # We maintain IDs for parameters that have already been freed,
                        # some additional parameters may take the same ID.
                        # In this case sequentially updating param_info messes up.
                        optim_param_index = optimizer.param_info["param2id"][
                            placeholder.param_id
                        ]
                        removes_from_param_info["param2id"].append(placeholder.param_id)
                        removes_from_param_info["id2param"].append(optim_param_index)
                        removes_from_param_info["param2shape"].append(
                            placeholder.param_id
                        )

                        adds_to_param_info["param2id"][id(p)] = optim_param_index
                        adds_to_param_info["id2param"][optim_param_index] = id(p)
                        adds_to_param_info["param2shape"][id(p)] = p.shape

                elif configuration_engine.rank in sender_ranks:
                    # Find the corresponding tensor parallel rank in ranks_need_layer
                    rank_need_layer = ranks_need_layer[
                        sender_ranks.index(configuration_engine.rank)
                    ]
                    for name, param in module.named_parameters():
                        buff = io.BytesIO()
                        # Find master weights
                        master_param = (
                            optimizer.working_to_master_map[param]
                            if isinstance(optimizer, HybridParallelAMPOptimizer)
                            else param
                        )

                        try:
                            # Pickle optimizer state
                            states = optimizer.optim.state[master_param]
                        except KeyError:
                            states = None

                        torch.save(
                            {
                                "states": states,
                                "parameter": master_param,
                            },
                            buff,
                        )
                        buff.seek(0)

                        tensor: torch.Tensor = torch.frombuffer(
                            buff.getbuffer(), dtype=torch.uint8
                        ).to(device)
                        size = torch.tensor(
                            [tensor.numel()], dtype=torch.int64, device=device
                        )

                        dist.send(size, rank_need_layer)
                        dist.send(tensor, rank_need_layer)

        # Update internal process group in ParallelModules
        my_layers = [
            layers[index] for index, has_layer in enumerate(new_my_layers) if has_layer
        ]
        tp_group = new_pg_mesh.get_group_along_axis(TP_AXIS)

        for layer in my_layers:
            module = layer_modules[layer]
            # Replace process group in ParallelModules

            for submodule in module.modules():
                if isinstance(submodule, ParallelModule):
                    for attr_name in dir(submodule):
                        attr = getattr(submodule, attr_name)
                        if isinstance(attr, dist.ProcessGroup):
                            logger.debug(
                                f"Replacing process group {attr} with {tp_group}"
                            )
                            setattr(submodule, attr_name, tp_group)

        for param_info_key, items in removes_from_param_info.items():
            for item in items:
                del optimizer.param_info[param_info_key][item]

        for param_info_key, items in adds_to_param_info.items():
            param_info: dict = optimizer.param_info[param_info_key]
            param_info.update(items)

        # free no-longer necessary parameter here
        # TODO: must create _buffer_placeholders and _parameter_placeholders
        old_layers = next(
            layers
            for ranks, layers in old_layers_per_ranks.items()
            if configuration_engine.rank in ranks
        )
        new_layers = next(
            layers
            for ranks, layers in new_layers_per_ranks.items()
            if configuration_engine.rank in ranks
        )
        for index, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
            if old_layer and not new_layer:
                module = layer_modules[layers[index]]
                ModelSharder.set_tensors_to_placeholder(module)

        self.set_pipelines(new_pipelines, new_num_microbatches)
        model, optimizer, _, dataloader, lr_scheduler = self.configure(
            model, optimizer, None, dataloader, lr_scheduler, forced=True
        )

        return model, optimizer, dataloader, lr_scheduler
