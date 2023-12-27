from __future__ import annotations
from oobleck.engine.arg_utils import DistArgs


class ConfigurationEngine:
    _instance: ConfigurationEngine = None

    def __init__(self):
        raise NotImplementedError(
            "Use get_instance() instead to get an instance of ConfigurationEngine."
        )

    @staticmethod
    def get_instance(*args, **kwargs) -> ConfigurationEngine:
        if ConfigurationEngine._instance is None:
            instance = ConfigurationEngine.__new__(ConfigurationEngine)

            # TODO: set initial attributes
            instance.rank_map: dict[str, list[int]] = None
            instance.rank: int = -1
            instance.local_rank: int = -1

            ConfigurationEngine._instance = instance

        return ConfigurationEngine._instance

    @property
    def is_master(self) -> bool:
        raise NotImplementedError()

    def get_distributed_information(self) -> DistArgs:
        # receive data from the agent

        self.rank_map: dict[str, list[int]] = {
            ip: list(
                range(
                    i * dist_args.tensor_parallel_size,
                    (i + 1) * dist_args.tensor_parallel_size,
                )
            )
            for i, ip in enumerate(dist_args.agent_ips)
        }
        self.rank = self.rank_map[my_ip][dist_args.local_rank]
        self.local_rank = dist_args.local_rank

    def send_distributed_port(self, port: int):
        pass

    def receive_distributed_port(self) -> int:
        pass
