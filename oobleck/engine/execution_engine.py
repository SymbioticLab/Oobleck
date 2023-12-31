from typing import Callable, Iterator

import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from oobleck_colossalai import HeterogeneousParallelPlugin
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.planning.pipeline_template import PipelineTemplate


class ExecutionEngine:
    """A main execution engine using an execution Backend."""

    def __init__(self, plugin: HeterogeneousParallelPlugin, **booster_kwargs):
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."

        self.plugin = plugin
        self.booster = Booster(plugin=plugin, **booster_kwargs)

        self.pipeline_templates: list[PipelineTemplate] | None = None

    def prepare(
        self,
        model: nn.Module,
        criterion: Callable | None = None,
        dataloader: DataLoader | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> tuple[nn.Module, Optimizer, Callable, LRScheduler, DataLoader]:
        """Initialize pipeline templates and distributed configuration."""

        if self.pipeline_templates is None:
            self.pipeline_templates = PipelineTemplate.generate_pipeline_templates(
                model
            )

        self._init_distributed()

        pipeline_instantiator = PipelineInstantiator(self.pipeline_templates)
        num_instances, num_microbatches = pipeline_instantiator.instantiate(
            dist.get_world_size()
        )
        self.plugin.set_pipeline_templates(num_instances, num_microbatches)
        return self.booster.boost(model, optimizer, criterion, dataloader, scheduler)

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
            configuration_engine.send_distributed_port(store.port)
            # this distributed port is broadcasted.
            # For master it is useless, so just discard it.
            configuration_engine.receive_distributed_port()
        else:
            port = configuration_engine.receive_distributed_port()
            store = dist.TCPStore(
                host_name=configuration_engine.dist_info[0].ip,
                port=port,
                world_size=configuration_engine.configuration_world_size,
                is_master=False,
                wait_for_workers=False,
            )
        dist.init_process_group(
            store=store,
            rank=configuration_engine.rank,
            world_size=configuration_engine.configuration_world_size,
        )

        assert dist.is_initialized(), "Distributed environment is not initialized."

    def execute(
        self,
        dataloader_iterator: Iterator,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        return_loss: bool = True,
        return_outputs: bool = False,
    ):
        self.booster.execute_pipeline(
            dataloader_iterator,
            model,
            criterion,
            optimizer,
            return_loss=return_loss,
            return_outputs=return_outputs,
        )

    # TODO (insujang): Implement the following
    # load_model, save_model, load_optimizer, save_optimizer, load_lr_scheduler, save_lr_scheduler
