import os
import redis
import deepspeed.comm as dist
import time

from ast import literal_eval
from typing import Dict, Tuple, List, Optional

from oobleck.utils.singleton import Singleton


class RedisReconfigurationMixin(object):
    def __init__(self):
        super().__init__()

    # def synchronize(self, num_ranks: int):
    #     """
    #     Wait until all ranks have called this function
    #     """
    #     key = "oobleck:synchronize"
    #     self.redis.incr(key)
    #     while int(self.redis.get(key)) < num_ranks:
    #         time.sleep(0.01)

    #     # Will only be deleted key is reached to the number of ranks
    #     self.redis.delete(key)

    def append_my_rank_to_layers(self, rank: int, layer_indices: List[int]):
        """
        Advertise that the rank has the layers.
        """
        with self.redis.pipeline() as pipe:
            for layer_index in layer_indices:
                pipe.rpush(f"oobleck:layer:{layer_index}", rank)

            pipe.execute()

    def get_all_having_layers(self) -> Dict[int, List[int]]:
        """
        Get all the lists of having layers.
        """
        keys = list(self.redis.scan_iter("oobleck:layer:*"))
        with self.redis.pipeline() as pipe:
            for key in keys:
                pipe.lrange(key, 0, -1)
            results = pipe.execute()

        return {
            int(k.split(":")[-1]): [int(r) for r in result]
            for k, result in zip(keys, results)
        }

    def append_missing_layers(self, rank: int, layer_indices: List[int]):
        """
        This is to advertise that the rank is missing some layers.
        Other processes will use this information to decide whether to send the layers to the rank.
        """
        with self.redis.pipeline() as pipe:
            for layer_index in layer_indices:
                pipe.rpush(f"oobleck:missing_layer:{layer_index}", rank)
            pipe.execute()

    def get_all_missing_layers(self) -> Dict[int, List[int]]:
        """
        Get all the lists of missing layers.
        """
        keys = list(self.redis.scan_iter("oobleck:missing_layer:*"))
        with self.redis.pipeline() as pipe:
            for key in keys:
                pipe.lrange(key, 0, -1)
            results = pipe.execute()

        return {
            int(k.split(":")[-1]): [int(r) for r in result]
            for k, result in zip(keys, results)
        }

    def wait_for_missing_layer(self, missing_layers: List[int], rank: int):
        """
        Some process will consume an item from the list "oobleck:missing_layer:{layer_index}"
        and put a key "oobleck:send_to:{rank}". This function will wait until the key is created.
        Use BLPOP command to wait for the key.
        """

        keys = [f"oobleck:send_to:{rank}:{layer}" for layer in missing_layers]
        result = self.redis.blpop(keys, timeout=0)

    def send_layer_to_rank(self, src_rank: int, target_rank: int, layer_index: int):
        """
        Put a key "oobleck:send_to:{rank}" to notify the rank that it should receive the layer.
        """
        self.redis.rpush(f"oobleck:send_to:{target_rank}:{layer_index}", src_rank)


@Singleton
class RedisClient(RedisReconfigurationMixin):
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
        # Use mget instead
        result = self.redis.mget(
            "oobleck:epoch", "oobleck:step", "oobleck:consumed_samples"
        )
        return (int(result[0]), int(result[1]), int(result[2]))
        # For backup
        with self.redis.pipeline() as pipe:
            pipe.get("oobleck:epoch")
            pipe.get("oobleck:step")
            pipe.get("oobleck:consumed_samples")
            epoch, step, consumed_samples = pipe.execute()
            return (int(epoch), int(step), int(consumed_samples))

    # def set_pipeline_ranks(self, pipeline_id: int, pipeline_ranks: List[int]):
    #     self.redis.set(f"oobleck:pipeline{pipeline_id}_ranks", str(pipeline_ranks))

    # def get_pipeline_ranks(self, pipeline_id: int) -> List[int]:
    #     return literal_eval(self.redis.get(f"oobleck:pipeline{pipeline_id}_ranks"))

    # def get_all_pipeline_ranks(self) -> List[List[int]]:
    #     keys = self.redis.scan_iter("oobleck:pipeline*_ranks")
    #     values = self.redis.mget(keys)
    #     ranks: List[List[int]] = [literal_eval(v) for v in values]
    #     return ranks

    # def get_ranks_for_layer(self, index: int) -> List[int]:
    #     """
    #     Iterate over all pipelines and return the ranks that has the layer
    #     """
    #     my_rank = dist.get_rank()
    #     keys = self.redis.scan_iter("oobleck:pipeline*_ranks")
    #     values = self.redis.mget(keys)
    #     ranks: List[List[int]] = [literal_eval(v) for v in values]

    #     return [r[index] for r in ranks if r[index] != my_rank]

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
