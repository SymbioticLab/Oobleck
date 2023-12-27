import socket
from colossalai.booster import Booster
from oobleck_colossalai import HeterogeneousParallelPlugin
from oobleck.planning.pipeline_template import PipelineTemplate
from oobleck.engine.arg_utils import DistArgs, TrainingArgs
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.engine.configuration_engine import ConfigurationEngine

from typing import Callable, Iterator

import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class ExecutionEngine:
    """A main execution engine using an execution Backend."""

    def __init__(
        self,
        training_args: TrainingArgs,
    ):
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."

        self.training_args = training_args

    def prepare(
        self,
        model: nn.Module,
        criterion: Callable | None = None,
        dataloader: DataLoader | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> list[nn.Module | Optimizer | LRScheduler | DataLoader]:
        """Initialize pipeline templates and distributed configuration."""

        self.pipeline_templates = PipelineTemplate.generate_pipeline_templates(model)
        self.pipeline_instantiator = PipelineInstantiator(self.pipeline_templates)

        dist_args = self._init_distributed()
        self.booster = self._init_colossalai_backend(dist_args)
        return self.booster.boost(model, optimizer, criterion, dataloader, scheduler)

    def _init_distributed(self) -> DistArgs:
        if dist.is_initialized():
            # Destroy all process group
            # TODO: if we try to destroy a process group where some operation is stuck,
            # destroying it might be stuck as well.
            # If this is witnessed, change it to destryoing all process groups
            # manually gathered in ThreadPoolExecutor.
            dist.destroy_process_group(dist.group.WORLD)

        configuration_engine = ConfigurationEngine.get_instance()
        dist_args = configuration_engine.get_distributed_information()

        my_ip: str = socket.gethostbyname(socket.gethostname())
        assert my_ip in dist_args.agent_ips, f"My IP {my_ip} is not in agent IPs."

        is_master = (
            next(iter(dist_args.agent_ips)) == my_ip and dist_args.local_rank == 0
        )

        if is_master:
            store = dist.TCPStore(
                host_name=my_ip,
                port=0,
                world_size=dist_args.world_size,
                is_master=True,
                wait_for_workers=False,
            )
            configuration_engine.send_distributed_port(store.port)
            # this distributed port is broadcast.
            # For master it is useless, so just discard it.
            configuration_engine.receive_distributed_port()
        else:
            port = configuration_engine.receive_distributed_port()
            store = dist.TCPStore(
                host_name=my_ip,
                port=port,
                world_size=dist_args.world_size,
                is_master=False,
                wait_for_workers=False,
            )
        dist.init_process_group(
            backend=dist_args.backend,
            store=store,
            rank=self.rank,
            world_size=dist_args.world_size,
        )

        assert dist.is_initialized(), "Distributed environment is not initialized."
        return dist_args

    def _init_colossalai_backend(self, dist_args: DistArgs) -> Booster:
        """Initialize ColossalAI backend."""
        plugin = HeterogeneousParallelPlugin(
            tp_size=dist_args.tensor_parallel_size,
            microbatch_size=self.training_args.microbatch_size,
        )

        num_instances, num_microbatches = self.pipeline_instantiator.instantiate(
            dist_args.world_size
        )
        plugin.set_pipeline_templates(num_instances, num_microbatches)

        return Booster(
            plugin=plugin, mixed_precision=self.training_args.mixed_precision
        )

    def execute(
        self,
        dataloader_iterator: Iterator,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
    ):
        self.booster.execute_pipeline(
            dataloader_iterator, model, criterion, optimizer, return_loss=True
        )
