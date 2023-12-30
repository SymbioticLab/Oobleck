import multiprocessing
import os
import pickle
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess

import grpc
import simple_parsing
from google.protobuf.empty_pb2 import Empty
from loguru import logger

from oobleck.elastic.master_service_pb2 import PortInfo
from oobleck.elastic.master_service_pb2_grpc import OobleckMasterStub
from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine


@dataclass
class OobleckAgentArguments:
    master_ip: str
    master_port: int
    agent_index: int


@dataclass
class Worker:
    pipe: Connection
    process: SpawnProcess

    @staticmethod
    def worker_main(
        pipe: Connection,
        gpu_index: int,
        code: str,
    ):
        """
        Worker process main function.

        It creates ConfigurationEngine that will internally be used in
        ExecutionEngine, and execute the given code.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        assert ConfigurationEngine._instance is None, (
            "ConfigurationEngine must not be initialized before "
            "worker_main() is called."
        )

        ConfigurationEngine.create(pipe, gpu_index)

        ccode = compile(code, "<string>", "exec")
        exec(ccode)


class Agent:
    """Oobleck Agent class.

    For each node, there is one agent process that manages
    worker processes in the node.
    """

    def __init__(self, agent_index: int, stub: OobleckMasterStub):
        self.agent_index = agent_index
        self.stub = stub

        # Get distributed information and code from the master
        dist_info = stub.GetDistInfo(Empty())
        self.dist_info = list(
            HostInfo(host.ip, host.slots, host.port) for host in dist_info.hosts
        )
        self.code = pickle.loads(stub.GetCode(Empty()).code)
        self.workers: list[Worker] = []

    def notify_reconfiguration_to_workers(self, dist_info: list[HostInfo]):
        for worker in self.workers:
            worker.pipe.send(dist_info)

    def watch_reconfiguration_notification(self):
        for dist_info in stub.WatchReconfigurationNotification(Empty()):
            self.notify_reconfiguration_to_workers(dist_info)

    def run_profiler(self):
        raise NotImplementedError()

    def launch_workers(self):
        """Launch worker processes.

        Before launching workers, check if profile data exists
        in this node. If not, call run_profiler() to collect
        profile data.
        TODO (insujang): implement run_profiler()
        """
        ctx: SpawnContext = multiprocessing.get_context("spawn")

        tensor_parallel_size = self.dist_info[0].slots
        ranks = range(
            self.agent_index * tensor_parallel_size,
            (self.agent_index + 1) * tensor_parallel_size,
        )

        for gpu_index, rank in enumerate(ranks):
            logger.info(f"Launching worker {rank} (GPU: {gpu_index})...")

            pipe, child_pipe = ctx.Pipe()

            process: SpawnProcess = ctx.Process(
                target=Worker.worker_main,
                args=(
                    child_pipe,
                    self.agent_index,
                    gpu_index,
                ),
                daemon=True,
            )
            process.start()
            self.workers.append(Worker(pipe, process))
            pipe.send(self.dist_info)

        # If this is the first agent, it should forward the master rank port
        if self.agent_index == 0:
            port: int = pipe.recv()
            self.stub.SetMasterRankPort(PortInfo(port=port))

        # Forward master port to all workers.
        self.forward_master_port()

    def forward_master_port(self):
        """
        Forward master port after receiving it from the master
        to all worker processes.
        """
        # Get master rank port from the master.
        # port will be 0 until master port is set.
        #
        port: int = 0
        while port == 0:
            time.sleep(0.1)
            port = self.stub.GetMasterRankPort(Empty()).port

        for worker in self.workers:
            worker.pipe.send(port)


if __name__ == "__main__":
    args: OobleckAgentArguments = simple_parsing.parse(
        OobleckAgentArguments, dest="agent"
    )

    # Connect to the master
    channel = grpc.insecure_channel(f"{args.agent.master_ip}:{args.agent.master_port}")
    stub = OobleckMasterStub(channel)

    agent = Agent(args.agent_index, stub)
    agent.launch_workers()
