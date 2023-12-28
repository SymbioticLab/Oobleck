import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
import socket
from types import CodeType

import os
import rpyc
import simple_parsing
from loguru import logger
from rpyc.core import brine

from oobleck.arg_utils import DistArgs
from oobleck.engine.configuration_engine import ConfigurationEngine


@rpyc.service
class AgentService(rpyc.Service):
    @rpyc.exposed
    def reconfigure(self, dist_info):
        pass


@dataclass
class OobleckAgentArguments:
    master_ip: str
    master_port: int
    agent_index: int
    gpu_indices: list[int]


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
        assert ConfigurationEngine._instance is None, (
            "ConfigurationEngine must not be initialized before "
            "worker_main() is called."
        )

        ConfigurationEngine.create(pipe, gpu_index)

        ccode = compile(code, "<string>", "exec")
        exec(ccode)


class Agent:
    def __init__(
        self,
        master_conn: rpyc.Connection,
        agent_args: OobleckAgentArguments,
        dist_args: DistArgs,
        code: str,
    ):
        self.master_conn = master_conn
        self.agent_args = agent_args
        self.dist_args = dist_args
        self.code = code
        self._workers: list[Worker] = []

    def launch_workers(self):
        """Launch worker processes.

        Before launching workers, test if profile data exists
        in this node. If not, call run_profiler() to collect
        profile data.
        """
        ctx: SpawnContext = multiprocessing.get_context("spawn")

        assert (
            len(self.dist_args.agent_ips) * self.dist_args.tensor_parallel_size
            == self.dist_args.world_size
        ), (
            f"Number of agents ({len(self.dist_args.agent_ips)}) "
            f"times tensor parallel size ({self.dist_args.tensor_parallel_size}) "
            f"must be equal to world size ({self.dist_args.world_size})."
        )

        assert (
            len(self.agent_args.gpu_indices) == self.dist_args.tensor_parallel_size
        ), (
            f"Number of GPUs ({len(self.agent_args.gpu_indices)}) "
            f"must be equal to tensor parallel size ({self.dist_args.tensor_parallel_size})."
        )

        ranks = range(
            self.agent_args.agent_index * self.dist_args.tensor_parallel_size,
            (self.agent_args.agent_index + 1) * self.dist_args.tensor_parallel_size,
        )

        for gpu_index, rank in zip(self.agent_args.gpu_indices, ranks):
            logger.info(f"Launching worker {rank} (GPU: {gpu_index})...")

            pipe, child_pipe = ctx.Pipe()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

            process: SpawnProcess = ctx.Process(
                target=Worker.worker_main,
                args=(
                    child_pipe,
                    gpu_index,
                ),
                daemon=True,
            )
            process.start()
            self._workers.append(Worker(pipe, process))
            pipe.send(dist_args)

        os.environ.pop("CUDA_VISIBLE_DEVICES")

        # If an agent has rank 0, it should forward its port to the master
        my_ip = socket.gethostbyname(socket.gethostname())
        if my_ip == dist_args.agent_ips[0]:
            self.forward_worker_port(self._workers[0].pipe)

    def forward_worker_port(self, pipe: Connection):
        port: int = pipe.recv()
        self.master.root.forward_rank0_port(port)

    def run_profiler(self):
        pass


if __name__ == "__main__":
    args: OobleckAgentArguments = simple_parsing.parse(
        OobleckAgentArguments, dest="agent"
    )

    args: OobleckAgentArguments = args.agent
    conn = rpyc.connect(args.agent.master_ip, args.agent.master_port)
    dist_args: DistArgs = conn.root.get_dist_info()
    code: str = conn.root.get_code()

    agent = Agent(conn, args, dist_args, code)
    agent.launch_workers()
