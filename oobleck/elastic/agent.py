import multiprocessing
import os
import runpy
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path

import grpc
import simple_parsing
import torch
from google.protobuf.empty_pb2 import Empty
from loguru import logger

from oobleck.elastic.master_service_pb2 import CodeInfo, DistInfo, PortInfo
from oobleck.elastic.master_service_pb2_grpc import OobleckMasterStub
from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine


@dataclass
class OobleckAgentArguments:
    master_ip: str
    master_port: int
    agent_index: int
    tag: str
    base_dir: Path


@dataclass
class Worker:
    pipe: Connection
    process: SpawnProcess

    @staticmethod
    def worker_main(
        pipe: Connection,
        agent_index: int,
        gpu_index: int,
        tag: str,
        base_dir: Path,
        script_path: Path,
        script_args: list[str],
    ):
        """
        Worker process main function.

        It creates ConfigurationEngine that will internally be used in
        ExecutionEngine, and execute the given code.
        """
        assert (
            torch.cuda.device_count() == 1
        ), "CUDA_VISIBLE_DEVICES must be set to a single GPU."

        assert ConfigurationEngine._instance is None, (
            "ConfigurationEngine must not be initialized before "
            "worker_main() is called."
        )

        ConfigurationEngine.create(pipe, agent_index, gpu_index, tag, base_dir)

        # Back up sys.argv and replace it with the given args
        original_argv = sys.argv.copy()
        sys.argv = [script_path.name] + script_args

        script_directory = str(script_path.parent)
        sys.path.insert(0, script_directory)

        runpy.run_path(script_path, run_name="__main__")

        # Restore sys.argv
        sys.argv = original_argv
        sys.path.remove(script_directory)


class Agent:
    """Oobleck Agent class.

    For each node, there is one agent process that manages
    worker processes in the node.
    """

    def __init__(
        self,
        agent_index: int,
        job_tag: str,
        base_dir: Path,
        stub: OobleckMasterStub,
    ):
        self.agent_index = agent_index
        self.tag = job_tag
        self.base_dir = base_dir
        self.stub = stub

        # Get distributed information and code from the master
        dist_info: DistInfo = stub.GetDistInfo(Empty())
        self.dist_info = list(
            HostInfo(host.ip, host.slots, host.port) for host in dist_info.hosts
        )
        training_args: CodeInfo = stub.GetCode(Empty())
        self.script: Path = Path(training_args.path)
        self.script_args: list[str] = [arg for arg in training_args.args]
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

        env_backup = os.environ.copy()

        for gpu_index, rank in enumerate(ranks):
            logger.info(f"Launching worker {rank} (GPU: {gpu_index})...")

            pipe, child_pipe = ctx.Pipe()

            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            process: SpawnProcess = ctx.Process(
                target=Worker.worker_main,
                args=(
                    child_pipe,
                    self.agent_index,
                    gpu_index,
                    self.tag,
                    self.base_dir,
                    self.script,
                    self.script_args,
                ),
                daemon=True,
            )
            process.start()
            self.workers.append(Worker(pipe, process))
            pipe.send(self.dist_info)

        os.environ = env_backup

        # If this is the first agent, it should forward the master rank port
        if self.agent_index == 0:
            logger.debug("Waiting for rank 0 port...")
            port: int = self.workers[0].pipe.recv()
            logger.debug(f"Received rank 0 port: {port}. Sending it to master.")
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

    def watch_worker_exit(self):
        """Watch worker exit and restart it.
        TODO: It must detect ANY worker exit, not just the first one."""
        for worker in self.workers:
            worker.process.join()


if __name__ == "__main__":
    args: OobleckAgentArguments = simple_parsing.parse(
        OobleckAgentArguments, dest="agent"
    )

    # Connect to the master
    channel = grpc.insecure_channel(f"{args.master_ip}:{args.master_port}")
    stub = OobleckMasterStub(channel)

    agent = Agent(args.agent_index, args.tag, args.base_dir, stub)
    agent.launch_workers()
    agent.watch_worker_exit()
