import itertools
import math
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.shardformer.policies.auto_policy import _fullname
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.dataloader import OobleckDataLoader
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.engine.plugin import OobleckPlugin
from oobleck.planner import create_pipeline_templates
from oobleck.profiler import ModelProfiler


class ExecutionEngine:
    """A main execution engine using an execution Backend.

    ExecutionEngine does not have a global view of distributed training.

    """

    def __init__(
        self,
        plugin: OobleckPlugin,
        **booster_kwargs,
    ):
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."
        assert isinstance(
            plugin, OobleckPlugin
        ), "Plugin must be an instance of OobleckPlugin."

        self.plugin = plugin

        configuration_engine = ConfigurationEngine.get_instance()
        self.tag = configuration_engine.tag
        self.base_dir = configuration_engine.base_dir
        self.booster = Booster(plugin=plugin, **booster_kwargs)

        self.pipeline_templates: dict[int, PipelineTemplate] | None = None

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
        dataloader: OobleckDataLoader | None = None,
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
                configuration_engine.world_size // self.plugin.shard_config["tp_size"]
            )

            self.pipeline_templates = create_pipeline_templates(
                _fullname(model),
                self.plugin.microbatch_size,
                list(range(min_num_nodes, max_num_nodes)) + [max_num_nodes],
                self.base_dir / self.tag / "profile",
            )

            logger.debug(f"Pipelines: {self.pipeline_templates}")

        configuration_engine.init_distributed()

        pipeline_instantiator = PipelineInstantiator(
            [
                self.pipeline_templates[num_nodes]
                for num_nodes in sorted(self.pipeline_templates.keys())
            ],
            self.plugin.global_batch_size // self.plugin.microbatch_size,
        )
        num_instances, num_microbatches = pipeline_instantiator.instantiate(
            len(configuration_engine.dist_info)
        )
        logger.debug(f"Pipeline instances: {num_instances}")
        logger.debug(f"Microbatches: {num_microbatches}")
        self.plugin.set_pipelines(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(template, num_templates)
                    for template, num_templates in num_instances.items()
                )
            ),
            num_microbatches,
        )
        return self.booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

    def _estimate_max_num_nodes_required(self):
        # TODO: implement it
        pass

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
