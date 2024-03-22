import copy
import gc
from collections import Counter

import torch
import torch.distributed as dist
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
from oobleck_colossalai.process_group_mesh import PP_AXIS
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

        # each tensor indicates which layers are held by each (new) rank
        old_layers_per_rank = [
            torch.zeros(num_layers, dtype=torch.bool, device="cuda")
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(old_layers_per_rank, old_my_layers)

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

        # copy missing layers

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
