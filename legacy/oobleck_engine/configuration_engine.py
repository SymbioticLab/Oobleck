from __future__ import annotations

import datetime
import itertools
import os
from multiprocessing.connection import Connection
from pathlib import Path

import torch.distributed as dist
from loguru import logger

from oobleck.elastic.run import HostInfo, HostStatus


class ConfigurationEngine:
    """
    An engine that manages internal configurations and distributed of Oobleck.
    This is the only engine in a worker process that communicates with an agent process.
    Communication with the agent is done via a pipe.
    Users should not touch this class manually.
    """

    _instance: ConfigurationEngine = None

    pipe: Connection
    agent_index: int
    tag: str
    base_dir: Path
    local_rank: int
    dist_info: list[HostInfo]
    rank_map: dict[HostInfo, list[int]]
    rank: int

    def __init__(self):
        raise NotImplementedError(
            "Use get_instance() instead to get an instance of ConfigurationEngine."
        )

    @staticmethod
    def create(
        pipe: Connection,
        agent_index: int,
        local_rank: int,
        tag: str,
        base_dir: Path,
    ) -> ConfigurationEngine:
        """Create a new instance of ConfigurationEngine."""
        if ConfigurationEngine._instance is not None:
            return ConfigurationEngine._instance

        instance = ConfigurationEngine.__new__(ConfigurationEngine)

        # TODO: set initial attributes.
        instance.pipe = pipe
        instance.agent_index = agent_index
        instance.tag = tag
        instance.base_dir = base_dir

        instance.local_rank = local_rank
        dist_info: list[HostInfo] = pipe.recv()
        instance.dist_info = dist_info

        logger.debug(f"dist_info: {dist_info}, agent_index: {agent_index}")

        instance.rank_map = {
            host: list(range(i * len(gpu_indices), (i + 1) * len(gpu_indices)))
            for i, host in enumerate(dist_info)
            if (gpu_indices := host.devices.split(","))
        }
        my_agent = dist_info[agent_index]
        instance.rank = instance.rank_map[my_agent][instance.local_rank]

        logger.debug(f"rank_map: {instance.rank_map}")

        ConfigurationEngine._instance = instance
        return ConfigurationEngine._instance

    @staticmethod
    def get_instance() -> ConfigurationEngine:
        assert (
            ConfigurationEngine._instance is not None
        ), "ConfigurationEngine is not initialized."
        return ConfigurationEngine._instance

    def get_host_update(self):
        """
        Get host update from the agent process.
        """
        my_agent = self.dist_info[self.agent_index]
        new_dist_info: list[HostInfo] = self.pipe.recv()

        assert not any(
            host.status == HostStatus.killed for host in new_dist_info
        ), "Worker should not see any killed agent status."

        agent_index = new_dist_info.index(my_agent)
        if new_dist_info[agent_index].status == HostStatus.terminating:
            # This process will be terminated after the iteration.
            logger.info(
                f"Worker {self.local_rank} in agent {self.agent_index} terminating..."
            )
            os._exit(0)

        self.dist_info = [
            host_info
            for host_info in new_dist_info
            if host_info.status == HostStatus.up
        ]
        self.agent_index = self.dist_info.index(my_agent)

        self.rank_map = {
            host: list(range(i * len(gpu_indices), (i + 1) * len(gpu_indices)))
            for i, host in enumerate(self.dist_info)
            if (gpu_indices := host.devices.split(","))
        }
        my_agent = self.dist_info[self.agent_index]
        self.rank = self.rank_map[my_agent][self.local_rank]

    @property
    def all_ranks(self) -> list[int]:
        return itertools.chain.from_iterable(self.rank_map.values())

    @property
    def world_size(self) -> int:
        if not self.dist_info:
            return 0

        num_devices = len(self.dist_info[0].devices.split(","))
        return num_devices * len(self.dist_info)

    @property
    def is_master(self) -> bool:
        """Return True if the current process is the master process (rank 0).
        Because it is not guarantted to have torch.distributed initialized,
        it is not possible to use dist.get_rank() == 0 to check.
        Instead, use agent index and local rank which are known without torch.ditributed.
        """
        return self.agent_index == 0 and self.local_rank == 0

    def recv_reconfiguration_notification(self) -> bool:
        """
        Block this thread until the agent sends a reconfiguration message.

        Returns:
            True if training requests immediate reconfiguration.
            False if some agents wll be terminated after the iteration.
        """
        try:
            message = self.pipe.recv()
            if message == "immediate_reconfigure":
                return True
            elif message == "reconfigure":
                return False

            raise ValueError(f"Unexpected reconfiguration message: {message}")
        except Exception:
            # Corresponding agent died. This process should also die.
            os._exit(1)

    def send_distributed_port(self, port: int):
        self.pipe.send(port)

    def receive_distributed_port(self) -> int:
        return self.pipe.recv()

    def init_distributed(self):
        """
        Initialize torch.distributed.

        When it was initialized before, destroy it first and reinitialize.
        When destruction happened, the function returns the old dist_info.
        """
        if dist.is_initialized():
            # TODO: if we try to destroy a process group where some operation is stuck,
            # destroying it might be stuck as well.
            # If this is witnessed, change it to destryoing all process groups
            # manually gathered in ThreadPoolExecutor.
            dist.destroy_process_group(dist.GroupMember.WORLD)

        assert not dist.is_initialized()

        if self.is_master:
            store = dist.TCPStore(
                host_name=self.dist_info[0].ip,
                port=0,
                world_size=self.world_size,
                is_master=True,
                wait_for_workers=False,
            )
            logger.debug(f"torch rank 0 port: {store.port}")
            self.send_distributed_port(store.port)
            # this distributed port is broadcasted and event this process receives it.
            # For master it is useless, so just discard it.
            self.receive_distributed_port()

            # Send anything to the agent back to ask to reset the port
            self.pipe.send(0)
        else:
            port = self.receive_distributed_port()
            logger.debug(f"Received torch.distributed rank 0 port: {port}")
            store = dist.TCPStore(
                host_name=self.dist_info[0].ip,
                port=port,
                world_size=self.world_size,
                is_master=False,
                wait_for_workers=False,
            )

        logger.debug(
            "Initializing torch.distributed. "
            f"rank: {self.rank}, world size: {self.world_size}"
        )

        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(minutes=5),
        )

        assert dist.is_initialized(), "Distributed environment is not initialized."
        logger.debug("Distributed environment initialized.")
