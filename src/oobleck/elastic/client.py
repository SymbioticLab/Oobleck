import os
import redis

from ast import literal_eval
from typing import Dict, Tuple, List, Optional


class ElasticWorkerRedisClientMixin(object):
    def __init__(self):
        super().__init__()

        redis_addr = os.environ["REDIS_ADDR"]

        self.redis = redis.Redis(redis_addr, 6379, decode_responses=True)
        assert self.redis.ping() == True

        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self.reconfiguration_thread: Optional[redis.client.PubSubWorkerThread] = None

        self.reconfiguration_info: List[
            Tuple[Dict[Tuple[str, int], List[int]], Tuple[str, int]]
        ] = []

    def __destory__(self):
        if self.reconfiguration_thread:
            self.pubsub.unsubscribe("oobleck:reconfiguration")
            self.reconfiguration_thread.stop()
        self.redis.close()

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
        return (first_node[0], 25400)

    def set_training_progress(self):
        pass

    def subscribe_reconfiguration(self):
        if self.reconfiguration_thread:
            return

        self.pubsub.subscribe(
            **{
                "oobleck:reconfiguration": lambda m: self.on_reconfiguration_requested(
                    m
                )
            }
        )
        self.reconfiguration_thread = self.pubsub.run_in_thread(
            sleep_time=0.01, daemon=True
        )

    def on_reconfiguration_requested(self, message):
        """
        We lose some some GPU, thus topology reconfiguration is needed.
        Pause training, reconfigure topology, and resume training.
        """

        print(f"Reconfiguration requested: {message}", flush=True)
        new_world_info = self.get_world_info()
        new_torch_master_info = self.get_torch_master_info(new_world_info)

        self.reconfiguration_info.append((new_world_info, new_torch_master_info))
