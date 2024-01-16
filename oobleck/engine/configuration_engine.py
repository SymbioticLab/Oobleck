from __future__ import annotations

from multiprocessing.connection import Connection
from pathlib import Path

from loguru import logger

from oobleck.elastic.run import HostInfo


class ConfigurationEngine:
    """
    An engine that manages internal configurations and distributed of Oobleck.
    Users should not touch this class manually.
    """

    _instance: ConfigurationEngine = None

    pipe: Connection
    agent_index: int
    tag: str
    base_dir: Path
    local_rank: int
    dist_info: list[HostInfo]
    rank_map: dict[str, list[int]]
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
        instance.pipe: Connection = pipe
        instance.agent_index = agent_index
        instance.tag = tag
        instance.base_dir = base_dir

        instance.local_rank: int = local_rank
        dist_info: list[HostInfo] = pipe.recv()
        instance.dist_info = dist_info

        logger.debug(f"dist_info: {dist_info}")

        instance.rank_map: dict[str, list[int]] = {
            f"{host.ip}:{host.port}": list(range(i * host.slots, (i + 1) * host.slots))
            for i, host in enumerate(dist_info)
        }
        my_ip = f"{dist_info[agent_index].ip}:{dist_info[agent_index].port}"
        instance.rank = instance.rank_map[my_ip][instance.local_rank]

        logger.debug(f"rank_map: {instance.rank_map}")

        ConfigurationEngine._instance = instance
        return ConfigurationEngine._instance

    @staticmethod
    def get_instance() -> ConfigurationEngine:
        assert (
            ConfigurationEngine._instance is not None
        ), "ConfigurationEngine is not initialized."
        return ConfigurationEngine._instance

    @property
    def configuration_world_size(self) -> int:
        return sum(host.slots for host in self.dist_info)

    @property
    def is_master(self) -> bool:
        return self.agent_index == 0 and self.local_rank == 0

    def send_distributed_port(self, port: int):
        self.pipe.send(port)

    def receive_distributed_port(self) -> int:
        return self.pipe.recv()
