import os
import redis
import deepspeed.comm as dist
import time

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

        self.reconfiguration_required = True

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

    def set_torch_master_info(
        self, addr: Optional[str] = None, port: Optional[int] = None
    ):
        if addr is None:
            self.redis.delete("oobleck:torch_master_info")
        else:
            self.redis.set("oobleck:torch_master_info", str((addr, port)))

    def get_torch_master_info(self) -> Tuple[str, int]:
        result = None
        while result is None:
            result = self.redis.get("oobleck:torch_master_info")
            time.sleep(0.01)
        return literal_eval(result)

    def set_training_progress(self, epoch: int, step: int, consumed_samples: int):
        if dist.get_rank() != 0:
            return

        with self.redis.pipeline() as pipe:
            pipe.set("oobleck:epoch", epoch)
            pipe.set("oobleck:step", step)
            pipe.set(
                "oobleck:consumed_samples",
                consumed_samples,
            )
            pipe.execute()

    def get_training_progress(self) -> Tuple[int, int, int]:
        with self.redis.pipeline() as pipe:
            pipe.get("oobleck:epoch")
            pipe.get("oobleck:step")
            pipe.get("oobleck:consumed_samples")
            epoch, step, consumed_samples = pipe.execute()
            return (int(epoch), int(step), int(consumed_samples))

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
        self.reconfiguration_required = True
