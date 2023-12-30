from __future__ import annotations

from multiprocessing.connection import Connection
from oobleck.elastic.run import HostInfo


class ConfigurationEngine:
    _instance: ConfigurationEngine = None

    def __init__(self):
        raise NotImplementedError(
            "Use get_instance() instead to get an instance of ConfigurationEngine."
        )

    @staticmethod
    def create(
        pipe: Connection,
        agent_index: int,
        local_rank: int,
    ) -> ConfigurationEngine:
        """Create a new instance of ConfigurationEngine."""
        if ConfigurationEngine._instance is not None:
            return ConfigurationEngine._instance

        instance = ConfigurationEngine.__new__(ConfigurationEngine)

        # TODO: set initial attributes.
        instance.pipe: Connection = pipe
        instance.agent_index = agent_index
        instance.local_rank: int = local_rank
        dist_info: list[HostInfo] = pipe.recv()
        instance.dist_info = dist_info

        instance.rank_map: dict[str, list[int]] = {
            f"{host.ip}:{host.port}": list(range(i * host.slots, (i + 1) * host.slots))
            for i, host in enumerate(dist_info)
        }
        my_ip = f"{dist_info[agent_index].ip}:{dist_info[agent_index].port}"
        instance.rank = instance.rank_map[my_ip][instance.local_rank]

        ConfigurationEngine._instance = instance
        return ConfigurationEngine._instance

    @staticmethod
    def get_instance() -> ConfigurationEngine:
        assert (
            ConfigurationEngine._instance is not None
        ), "ConfigurationEngine is not initialized."
        return ConfigurationEngine._instance

    @property
    def is_master(self) -> bool:
        return self.agent_index == 0 and self.local_rank == 0

    def send_distributed_port(self, port: int):
        self.pipe.send(port)

    def receive_distributed_port(self) -> int:
        return self.pipe.recv()
