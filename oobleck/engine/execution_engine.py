import math
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from loguru import logger
from oobleck_colossalai import HeterogeneousDataLoader, HeterogeneousParallelPlugin
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.planner import create_pipeline_templates
from oobleck.profiler import ModelProfiler


class ExecutionEngine:
    """A main execution engine using an execution Backend."""

    def __init__(
        self,
        plugin: HeterogeneousParallelPlugin,
        **booster_kwargs,
    ):
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."

        self.plugin = plugin

        configuration_engine = ConfigurationEngine.get_instance()
        self.tag = configuration_engine.tag
        self.base_dir = configuration_engine.base_dir
        self.booster = Booster(plugin=plugin, **booster_kwargs)

        self.pipeline_templates: list[PipelineTemplate] | None = None

    @property
    def is_master(self) -> bool:
        configuration_engine = ConfigurationEngine.get_instance()
        if configuration_engine is None:
            raise RuntimeError("ConfigurationEngine must be initialized.")
        return configuration_engine.is_master

    def prepare(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        criterion: Callable | None = None,
        dataloader: HeterogeneousDataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> tuple[nn.Module, Optimizer, Callable, DataLoader, LRScheduler]:
        """Initialize pipeline templates and distributed configuration."""

        configuration_engine = ConfigurationEngine.get_instance()

        if self.pipeline_templates is None:
            logger.debug("Creating pipeline templates...")
            profiler = ModelProfiler(
                self.tag,
                model.__class__.__name__,
                model.config,
                self.plugin.microbatch_size,
                self.base_dir,
            )

            # Check profile data exists
            if not profiler.profile_exists():
                logger.debug("Profile does not exist. Start profiling.")
                profile_dataloder = DataLoader(
                    dataloader.dataset, batch_size=self.plugin.microbatch_size
                )
                inputs = next(iter(profile_dataloder))
                profiler.profile(
                    configuration_engine.local_rank,
                    self.plugin.shard_config["tp_size"],
                    inputs,
                )

            # Calculate the minimum number of nodes required
            memory = torch.cuda.get_device_properties(0).total_memory
            min_num_nodes = max(
                1,
                math.ceil(profiler.mem_consumption / memory),
            )
            max_num_nodes = (
                configuration_engine.configuration_world_size
                // self.plugin.shard_config["tp_size"]
            )

            self.pipeline_templates = create_pipeline_templates(
                self.plugin.microbatch_size,
                list(range(min_num_nodes, max_num_nodes)) + [max_num_nodes],
                self.base_dir / self.tag / "profile",
            )

            logger.debug(f"Pipelines: {self.pipeline_templates}")

        self._init_distributed()

        pipeline_instantiator = PipelineInstantiator(
            [
                self.pipeline_templates[num_nodes]
                for num_nodes in sorted(self.pipeline_templates.keys())
            ],
            self.plugin.global_batch_size,
        )
        num_instances, num_microbatches = pipeline_instantiator.instantiate(
            len(configuration_engine.dist_info)
        )
        logger.debug(f"Pipeline instances: {num_instances}")
        logger.debug(f"Microbatches: {num_microbatches}")
        self.plugin.set_pipeline_templates(num_instances, num_microbatches)
        return self.booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

    def _init_distributed(self):
        if dist.is_initialized():
            # Destroy all process group
            # TODO: if we try to destroy a process group where some operation is stuck,
            # destroying it might be stuck as well.
            # If this is witnessed, change it to destryoing all process groups
            # manually gathered in ThreadPoolExecutor.
            dist.destroy_process_group(dist.group.WORLD)

        # ConfigurationEngine must be initialized by worker_main().
        configuration_engine = ConfigurationEngine.get_instance()

        if configuration_engine.is_master:
            store = dist.TCPStore(
                host_name=configuration_engine.dist_info[0].ip,
                port=0,
                world_size=configuration_engine.configuration_world_size,
                is_master=True,
                wait_for_workers=False,
            )
            logger.debug(f"torch rank 0 port: {store.port}")
            configuration_engine.send_distributed_port(store.port)
            # this distributed port is broadcasted.
            # For master it is useless, so just discard it.
            configuration_engine.receive_distributed_port()
        else:
            port = configuration_engine.receive_distributed_port()
            logger.debug(f"Received torch rank 0 port: {port}")
            store = dist.TCPStore(
                host_name=configuration_engine.dist_info[0].ip,
                port=port,
                world_size=configuration_engine.configuration_world_size,
                is_master=False,
                wait_for_workers=False,
            )
        logger.debug(
            "Initializing torch.distributed. "
            f"rank: {configuration_engine.rank}, world size: {configuration_engine.configuration_world_size}"
        )

        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=configuration_engine.rank,
            world_size=configuration_engine.configuration_world_size,
        )

        logger.debug("Distributed environment initialized.")

        assert dist.is_initialized(), "Distributed environment is not initialized."

    def execute(
        self,
        dataloader_iterator: Iterator,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict[str, Any]:
        return self.booster.execute_pipeline(
            dataloader_iterator,
            model,
            criterion,
            optimizer,
            return_loss=return_loss,
            return_outputs=return_outputs,
        )

    # TODO (insujang): Implement the following
    # load_model, save_model, load_optimizer, save_optimizer, load_lr_scheduler, save_lr_scheduler
