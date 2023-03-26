import redis
import rpyc

from enum import Enum
from rpyc.utils.server import ThreadedServer
from typing import Any, List, Dict, Tuple, Optional, TypeVar

from deepspeed.utils import logging

OOBLECK_MASTER_DEFAULT_PORT = 27322


class Reconfiguration(Enum):
    NEXT_ITERATION = 0
    IMMEDIATELY = 1
    NONE = 2


T = TypeVar("T", bound="OobleckMaster")


class MasterServiceMixin(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.connections: List[Tuple(int, Tuple[str, int])] = []

    def on_connect(self: T, conn: rpyc.Connection):
        conn_id = (id(conn), conn._channel.stream.sock.getpeername())
        logger.info("Adding connection %s", conn_id)
        self.connections.append(conn_id)

    def on_disconnect(self: T, conn: rpyc.Connection):
        # Remove the disconnected node from the connection list
        # Finding a connection is done by id(conn)
        conn_id = next(filter(lambda c: c[0] == id(conn), self.connections), None)
        assert conn_id is not None, "Non managed ID is disconnected..."
        logger.info("A connection %s disconnected", conn_id)

        # Remove connection
        self.connections.remove(conn_id)

        # If it is an agent, remove it from the agent list too
        self.delete_agent(conn_id[0])

    def exposed_register_agent(self: T, agent_ip: Tuple[str, int]):
        # Since it is not possible to get Connection ID in RPC, we extract it from Connection list with its IP address.
        conn_id = next(filter(lambda c: c[1] == agent_ip, self.connections), None)
        assert conn_id, "No connection with this client."

        self.add_agent(conn_id[0], agent_ip)

    def exposed_run_model(
        self: T,
        fault_tolerance_spec: int,
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        if self.training_in_progress:
            raise Exception("Training already in progress")

        if len(self.agents) == 0:
            raise RuntimeError("No agent online.")

        # all five arguments for OobleckEngine
        execution_info = {
            "fault_tolerance_spec": fault_tolerance_spec,
            "model_name": model_name,
            "model_args": model_args,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
        }

        # Replcae the above etcd transaction to one redis transaction using pipeline.
        with self.redis.pipeline() as pipe:
            pipe.set("oobleck:world_info", str(self.world_info))
            pipe.set("oobleck:execution_info", str(execution_info))
            pipe.set("oobleck:reconfiguration", Reconfiguration.NONE.name)
            pipe.set("oobleck:epoch", 0)
            pipe.set("oobleck:step", 0)
            pipe.set("oobleck:consumed_samples", 0)
            pipe.publish("oobleck:training_start", 0)
            pipe.execute()

        self.training_in_progress = True
        logger.info("Training %s started.", model_name)


class OobleckMaster(MasterServiceMixin):
    def __init__(
        self,
        num_gpus_per_node: int,
        ip: Optional[str] = None,
        port: int = OOBLECK_MASTER_DEFAULT_PORT,
    ):
        super().__init__()

        self.ip = ip
        self.port = port
        self.num_gpus_per_node = num_gpus_per_node
        self.training_in_progress = False

        self.agents: Dict[int, Tuple[str, int]] = {}

        self.server = ThreadedServer(self, ip, port=port)
        self.redis = redis.Redis(host=ip, port=6379, decode_responses=True)
        for key in self.redis.scan_iter("oobleck:*"):
            self.redis.delete(key)
        self.redis.set("oobleck:reconfiguration", Reconfiguration.NONE.name)

    def start(self):
        logger.info(f"Serving on {self.server.host}:{self.server.port}")
        self.server.start()

    @property
    def world_info(self) -> str:
        results: Dict[Tuple[str, int], List[int]] = {}
        num_gpus_used = 0
        for agent in self.agents.values():
            results[agent] = list(
                range(num_gpus_used, num_gpus_used + self.num_gpus_per_node)
            )
            num_gpus_used += self.num_gpus_per_node

        return str(results)

    def add_agent(self, conn_id: int, agent_id: Tuple[str, int]):
        logger.info("Adding agent %s", agent_id)
        if conn_id in self.agents:
            return
        self.agents[conn_id] = agent_id

        # Change world info
        # Notify workers that reconfiguration is needed (at the next iteration)
        self._broadcast_world_info_change(Reconfiguration.NEXT_ITERATION)

    def delete_agent(self, conn_id: int):
        try:
            agent_id = self.agents[conn_id]
            logger.info("Removing agent %s", agent_id)
            del self.agents[conn_id]

            # Change world info
            # Notify workers that reconfiguration is needed immediately
            self._broadcast_world_info_change(Reconfiguration.IMMEDIATELY)

            if self.training_in_progress and len(self.agents) == 0:
                # No more agents. If training was in progress, cancel it.
                logger.warning("No agent online. Cancel training.")
                self.training_in_progress = False

        except KeyError:
            pass

    def _broadcast_world_info_change(self, reconfiguration: Reconfiguration):
        with self.redis.pipeline() as pipe:
            pipe.multi()
            pipe.set("oobleck:world_info", str(self.world_info))
            pipe.set("oobleck:reconfiguration", reconfiguration.name)
            pipe.execute()


if __name__ == "__main__":
    logger = logging.LoggerFactory.create_logger("oobleck.master")

    master = OobleckMaster(1, "127.0.0.1")
    master.start()
