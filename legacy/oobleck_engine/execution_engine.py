import itertools
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.shardformer.policies.auto_policy import _fullname
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.auto_policy import get_autopolicy
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)
from loguru import logger
from torch.distributed import distributed_c10d
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.engine.plugin import OobleckPlugin
from oobleck.planning.planner import create_pipeline_templates
from oobleck.planning.profiler import ModelProfiler


class ExecutionEngine:
    """A main execution engine using an execution Backend.

    ExecutionEngine does not have a global view of distributed training.

    """

    def __init__(
        self,
        plugin: OobleckPlugin,
        **booster_kwargs,
    ):
        assert os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", None) == "0", (
            "Please set TORCH_NCCL_ASYNC_ERROR_HANDLING=0. "
            "Otherwise it is not possible to handle NCCL failure."
        )
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."
        assert isinstance(
            plugin, OobleckPlugin
        ), "Plugin must be an instance of OobleckPlugin."

        self.plugin = plugin

        self.pipeline_templates: dict[int, PipelineTemplate] | None = None
        self.booster: Booster | None = None
        self.booster_kwargs = booster_kwargs

        self.notification_receiver_thread: Thread | None = None
        self.need_reconfiguration: bool = False

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
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> tuple[nn.Module, Optimizer, Callable, DataLoader, LRScheduler]:
        """Initialize pipeline templates and distributed configuration.

        This function automatically initializes torch.distributed,
        create pipeline templates, instantiate pipelines, and then boost the model.
        """

        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."

        if self.pipeline_templates is not None:
            raise RuntimeError(
                "Pipeline templates are already initialized. "
                "You should not call prepare() more than once."
            )

        configuration_engine = ConfigurationEngine.get_instance()

        profiler = ModelProfiler(
            configuration_engine.tag,
            model_name_or_path=_fullname(model),
            optimizer_class=_fullname(optimizer),
            config=model.config,
            precision=self.plugin.precision,
            tp_size=self.plugin.tp_size,
            base_dir=configuration_engine.base_dir,
        )

        profile_dataloder = DataLoader(
            dataloader.dataset,
            batch_size=self.plugin.microbatch_size,
            collate_fn=dataloader.collate_fn,
        )
        inputs = next(iter(profile_dataloder))
        profiler.init_profile(inputs)

        configuration_engine.init_distributed()
        profile_data = profiler.load_profile(self.plugin.microbatch_size)

        # Calculate the minimum number of nodes required
        memory = torch.cuda.get_device_properties(0).total_memory
        min_num_nodes = max(
            1,
            math.ceil(sum(layer.mem_required for layer in profile_data) / memory),
        )
        max_num_nodes = configuration_engine.world_size // self.plugin.tp_size

        logger.debug("Creating pipeline templates...")
        model_name = _fullname(model)
        pipeline_templates = create_pipeline_templates(
            model_name,
            profile_data,
            list(range(min_num_nodes, max_num_nodes + 1)),
        )

        policy: PipelineTemplatePolicyBase = get_autopolicy(model_name)
        policy.set_model(model)
        for key in list(pipeline_templates.keys()):
            try:
                template = pipeline_templates[key]
                policy.pipeline_template_sanity_check(template)
            except ValueError as e:
                logger.debug(
                    f"Pipeline template {template} failed to pass sanity check and removed: {e}"
                )
                del pipeline_templates[key]

        if not pipeline_templates:
            raise RuntimeError("No pipeline templates created.")

        self.pipeline_templates = pipeline_templates
        logger.debug(f"Pipeline templates: {self.pipeline_templates}")

        pipeline_instantiator = PipelineInstantiator(
            self.pipeline_templates,
            self.plugin.global_batch_size // self.plugin.microbatch_size,
            self.plugin.fault_tolerance_threshold,
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
        self.booster = Booster(plugin=self.plugin, **self.booster_kwargs)
        return self.booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

    def _estimate_max_num_nodes_required(self):
        # TODO: implement it
        pass

    def notification_receive_func(self):
        logger.info("Start failure notification watcher.")
        configuration_engine = ConfigurationEngine.get_instance()
        immediate_restart = configuration_engine.recv_reconfiguration_notification()
        self.need_reconfiguration = True

        if immediate_restart:
            logger.warning(
                "Immediate reconfiguration is needed. Termiating torch.distributed."
            )
            self.on_receive_reconfiguration_notifiation()
        else:
            logger.info(
                "Received reconfiguration request. Will start after the current iteration."
            )

    def on_receive_reconfiguration_notifiation(self):
        """
        A failure event is received from any worker.
        The reconfiguration engine should reconfigure affected pipelines
        using the set of pipeline templates.
        This function is called in such a case.
        """
        logger.warning("Reconfiguration: shutting down all process groups.")

        device = get_accelerator().get_current_device()
        all_pgs = list(distributed_c10d._pg_map.keys())
        all_pgs = [pg._get_backend(device) for pg in all_pgs]
        all_pgs = [
            pg.wrapped_pg
            if isinstance(pg, distributed_c10d._ProcessGroupWrapper)
            else pg
            for pg in all_pgs
        ]

        with ThreadPoolExecutor(max_workers=len(all_pgs)) as executor:
            executor.map(lambda pg: pg._shutdown(), all_pgs)

        try:
            dist.destroy_process_group(dist.GroupMember.WORLD)
        except Exception:
            pass
        finally:
            # If distributed_c10d says GroupMember.WORLD is None
            # To make sure, clean up internal data structure again
            distributed_c10d._update_default_pg(None)
            distributed_c10d._world.pg_map.clear()
            distributed_c10d._world.pg_names.clear()
            distributed_c10d._world.pg_group_ranks.clear()
            distributed_c10d._world.pg_backend_config.clear()
            distributed_c10d._world.pg_to_tag.clear()
            distributed_c10d._world.tags_to_pg.clear()
            distributed_c10d._world.pg_coalesce_state.clear()
            distributed_c10d._world.pg_default_device.clear()
            distributed_c10d._world.group_count = 0

    def execute(
        self,
        dataloader_iterator: Iterator,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict[str, Any] | None:
        if self.need_reconfiguration:
            self.notification_receiver_thread.join()
            if dist.is_initialized():
                # If torch.distributed is still initialized, this means
                # the configuration was not immediate and should be done now.
                torch.cuda.synchronize()
                self.on_receive_reconfiguration_notifiation()
            return None

        if getattr(dataloader_iterator, "invalidated", False):
            raise RuntimeError(
                "The dataloader iterator has been invalidated. "
                "Please recreate the iterator before resuming training."
            )

        if (
            self.notification_receiver_thread is None
            or not self.notification_receiver_thread.is_alive()
        ):
            self.notification_receiver_thread = Thread(
                target=self.notification_receive_func,
                name="failure_notification_watcher",
                daemon=True,
            )
            self.notification_receiver_thread.start()

        try:
            return self.booster.execute_pipeline(
                dataloader_iterator,
                model,
                criterion,
                optimizer,
                return_loss=return_loss,
                return_outputs=return_outputs,
            )
        except dist.DistError as e:
            logger.warning(
                f"DistError caught: {str(e)}. Immediate reconfiguration is needed."
            )
            # Failure happens and WORLD process group has been destroyed.
            setattr(dataloader_iterator, "invalidated", True)

            return None

    def reconfigure(
        self, model: nn.Module, optimizer: Optimizer, dataloader: DataLoader
    ) -> tuple[nn.Module, Optimizer, DataLoader]:
        logger.info("Waiting for failure notification watcher to finish.")
        self.notification_receiver_thread.join()

        logger.info("Start reconfiguration in 5 seconds...")
        time.sleep(5)

        assert not dist.is_initialized()
        model, optimizer, dataloader, _ = self.plugin.reconfigure(
            self.pipeline_templates, model, optimizer, dataloader
        )

        self.need_reconfiguration = False
        logger.info("Reconfiguration is done.")
        return model, optimizer, dataloader
