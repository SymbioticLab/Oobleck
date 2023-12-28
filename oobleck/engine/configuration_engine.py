from __future__ import annotations

import socket
from multiprocessing.connection import Connection

from oobleck.arg_utils import DistArgs


class ConfigurationEngine:
    _instance: ConfigurationEngine = None

    def __init__(self):
        raise NotImplementedError(
            "Use get_instance() instead to get an instance of ConfigurationEngine."
        )

    @staticmethod
    def create(
        pipe: Connection,
        local_rank: int,
    ) -> ConfigurationEngine:
        """Create a new instance of ConfigurationEngine."""
        if ConfigurationEngine._instance is not None:
            return ConfigurationEngine._instance

        instance = ConfigurationEngine.__new__(ConfigurationEngine)

        # TODO: set initial attributes.
        instance.pipe: Connection = pipe
        instance.local_rank: int = local_rank
        instance.dist_args: DistArgs = pipe.recv()

        instance.rank_map: dict[str, list[int]] = {
            ip: list(
                range(
                    i * instance.dist_args.tensor_parallel_size,
                    (i + 1) * instance.dist_args.tensor_parallel_size,
                )
            )
            for i, ip in enumerate(instance.dist_args.agent_ips)
        }
        my_ip = socket.gethostbyname(socket.gethostname())
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
        return self.rank in next(iter(self.rank_map))[0]

    def send_distributed_port(self, port: int):
        self.pipe.send(port)

    def receive_distributed_port(self) -> int:
        port: int = self.pipe.recv()
        return port
