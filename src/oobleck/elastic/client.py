import os
import rpyc
import redis

from ast import literal_eval
from typing import Optional, Dict, List, Tuple


class ElasticWorkerClientMixin(object):
    """A mixin that is used by worker processes
    to query data required for training.
    Only communicate with etcd.
    """

    def __init__(self):
        super().__init__()
        redis_addr = os.environ["REDIS_ADDR"]

        self.redis = redis.Redis(redis_addr, 6379, decode_responses=True)
        assert self.redis.ping() == True

        self.pubsub = self.redis.pubsub()

    def get_world_info(self) -> Dict[Tuple[str, int], List[int]]:
        world_info: Dict[Tuple[str, int], List[int]] = literal_eval(
            self.redis.get("oobleck:world_info")
        )
        if len(world_info) == 0:
            return {}
        assert all(
            len(gpus) > 0 for gpus in world_info.values()
        ), "Some node has no GPUs."
        assert all(
            len(gpus) == len(next(iter(world_info.values())))
            for gpus in world_info.values()
        ), "Some node has different number of GPUs."

        return world_info

    def get_torch_master_info(
        self, world_info: Dict[Tuple[str, int], List[int]]
    ) -> Tuple[str, int]:
        first_node = next(iter(sorted(world_info)))
        return (first_node[0], "25400")

    def on_reconfiguration_requested(self):
        """
        We lose some some GPU, thus topology reconfiguration is needed.
        Pause training, reconfigure topology, and resume training.

        Args:
            event (_type_): an event of reconfiguration_needed
        """

        self.pubsub.subscribe("oobleck:reconfiguration")
