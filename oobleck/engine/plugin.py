from collections import Counter

from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    get_param_info,
)
from colossalai.interface import ModelWrapper, OptimizerWrapper
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.dataloader import OobleckDataLoader
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

    def prepare_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 1024,
        pin_memory: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> OobleckDataLoader:
        _kwargs = kwargs.copy()
        _kwargs.pop("sampler", None)
        _kwargs.pop("batch_sampler", None)

        return OobleckDataLoader(
            dataset,
            global_batch_size=self.global_batch_size,
            microbatch_size=self.microbatch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **_kwargs,
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
        dataloader: OobleckDataLoader,
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
        old_pg_mesh = self.pg_mesh
        old_rank_map = configuration_engine.rank_map
        old_pipelines = self.pipelines

        # Reset process group
        _, removed_hosts = configuration_engine.get_host_update()
        configuration_engine.init_distributed()

        num_hosts_per_pipeline = [pipeline.num_stages for pipeline in old_pipelines]
        removed_ranks_list = [old_rank_map[host] for host in removed_hosts]
        all_ranks = [rank for ranks in old_rank_map.values() for rank in ranks]
        for removed_ranks in removed_ranks_list:
            for pipeline_index, ranks_in_mesh in enumerate(old_pg_mesh.mesh):
                if removed_ranks in ranks_in_mesh:
                    num_hosts_per_pipeline[pipeline_index] -= 1

            all_ranks.remove(removed_ranks)

        # Create new pipelines.
        # TODO: If there is no feasible pipeline, merge pipelines
        new_pipelines = [
            pipeline_templates[num_stages] for num_stages in num_hosts_per_pipeline
        ]

        # Redistribute microbatches
        instantiator = PipelineInstantiator(new_pipelines, self.global_batch_size)
        _, num_microbatches = instantiator.distribute_batch(
            dict(Counter(new_pipelines))
        )

        self.set_pipelines(new_pipelines, num_microbatches)

        # create copy plan (with torch.distributed)

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
        dataloader.reconfigure()

        return model, optimizer, dataloader, lr_scheduler
