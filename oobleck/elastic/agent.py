import asyncio
import multiprocessing
import os
import signal
import socket
import sys
from dataclasses import dataclass
from multiprocessing import connection

import simple_parsing
from deepspeed.utils.logging import LoggerFactory

import oobleck.elastic.message_util as message_util
from oobleck.csrc.planning.pipeline_template import get_profile_results
from oobleck.elastic.training_util import OobleckAgentArguments, OobleckArguments
from oobleck.elastic.worker import worker_main
from oobleck.planning.profiler import profile

logger = LoggerFactory.create_logger("oobleck_agent")


@dataclass
class Worker:
    pipe: connection.Connection
    process: multiprocessing.Process


class OobleckAgent:
    """
    Oobleck agent process that runs on each agent node.
    It manages worker processes, where one worker is a rank in distributed training.

    An agent does:
    1. It registers itself to the master daemon when it starts.
    2. After registration, it periodically sends a liveness packet to the master,
       and wait for reconfiguration notification.
    3. Once reconfiguration is arrived, it sends a SIGUSR1 signal to all workers,
       letting them know that they need reconfiguration.
    4. An agent and workers have a dedicated mp.queue. After sending a signal,
       the agent queries a new distribution information from the master and forward it to workers.
    """

    def __init__(self, args: OobleckAgentArguments):
        self._args = args
        self._rank_map: dict[str, list[int]] = {
            ip: list(range(i * args.num_workers, (i + 1) * args.num_workers))
            for i, ip in enumerate(args.node_ips)
        }
        self._conn: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        self._workers: list[Worker] = []
        self._response_callbacks: dict[message_util.RequestType, callable] = {}

    async def run(self):
        await self._connect_to_master(self._args.master_ip, self._args.master_port)
        await self._register_agent()
        await self._launch_workers(self._args.num_workers, self._args.job_args)

    async def _connect_to_master(self, master_ip: str, master_port: int):
        # TODO: add timeout in connection
        self._conn = await asyncio.wait_for(
            asyncio.open_connection(master_ip, master_port),
            timeout=message_util.TIMEOUT,
        )

    async def _register_agent(self):
        await message_util.send_request_type(
            self._conn[1], message_util.RequestType.REGISTER_AGENT
        )
        result, req = await message_util.recv_response(self._conn[0])
        if (
            result is not message_util.Response.SUCCESS
            or req is not message_util.RequestType.REGISTER_AGENT
        ):
            raise ConnectionError("Failed to register agent")

        # When success, start pinging the master
        # asyncio.create_task(self.ping())
        asyncio.create_task(self.on_receive_response())

    def _run_profiler(self, num_workers: int, args: OobleckArguments):
        ctx = multiprocessing.get_context("spawn")
        profiler_processes: list[multiprocessing.Process] = []

        for index in range(num_workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
            my_ip = socket.gethostbyname(socket.gethostname())
            master_ip = self._args.node_ips[0]
            master_port = 23456
            world_size = len(self._args.node_ips) * num_workers
            rank = self._args.node_ips.index(my_ip) * num_workers + index
            process = ctx.Process(
                target=profile,
                args=(args, master_ip, master_port, num_workers, world_size, rank),
            )
            process.start()
            profiler_processes.append(process)

        for process in profiler_processes:
            process.join()

    async def _launch_workers(self, num_workers: int, args: OobleckArguments):
        # Test if profile data exists
        try:
            get_profile_results(args.model_name, args.model_tag, args.microbatch_size)
        except Exception:
            # Run profiler
            logger.warning(
                f"Profile data for model {args.model_name} not found. Launching profiler..."
            )
            self._run_profiler(num_workers, args)

        ctx = multiprocessing.get_context("spawn")
        for index in range(num_workers):
            logger.info(f"Launching worker {index}...")
            # TODO: add all arguments. Arguments should be passed from the master
            # via command line arguments.
            pipe, child_pipe = ctx.Pipe()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

            process = ctx.Process(
                target=worker_main,
                args=(
                    index,
                    len(self._args.node_ips) * num_workers,
                    1,
                    child_pipe,
                    args,
                ),
                daemon=True,
            )
            process.start()

            self._workers.append(Worker(pipe, process))

            # TODO: detect worker failure and report it to the master.
            # For now only consider node failure, thus training will go stuck
            # if a worker fails.

        os.environ.pop("CUDA_VISIBLE_DEVICES")

        # test
        for worker in self._workers:
            worker.process.join()

    async def forward_worker_port(self, pipe: connection.Connection):
        _, w = self._conn
        port: int = pipe.recv()
        await message_util.send_request_type(
            w, message_util.RequestType.FORWARD_RANK0_PORT
        )
        await message_util.send(w, port, need_pickle=True, drain=True, close=False)

    async def on_receive_worker_port(self):
        r, w = self._conn
        port: int = await message_util.recv(r, need_pickle=True)
        for worker in self._workers:
            worker.pipe.send(port)

    async def send_request(
        self,
        request: message_util.RequestType,
        args: dict | None = None,
        callback: callable = None,
    ):
        if request in self._response_callbacks:
            logger.warning(
                f"Already pending request for the same request type {request}"
            )
            return

        if request is not message_util.RequestType.PING:
            self._response_callbacks[request] = callback
        await message_util.send_request_type(self._conn[1], request)

        if args is not None:
            await message_util.send(
                self._conn[1], args, need_pickle=True, drain=True, close=False
            )

    async def on_receive_reconfiguration(self):
        logger.debug("reconfiguration request received")
        lost_node: str = await message_util.recv(self._conn[0], need_pickle=True)

        # This is for emulating a lost node by sending a command from the master.
        # Won't happen in normal case.
        if lost_node == socket.gethostbyname(socket.gethostname()):
            logger.info("I'm the lost node. I'll terminate myself.")
            for worker in self._workers:
                worker.process.terminate()
            sys.exit(1)

        else:
            # Send SIGUSR1 signal to workers
            lost_ranks: list[int] = self._rank_map.pop(lost_node)
            for worker in self._workers:
                worker.pipe.send(lost_ranks)
                os.kill(worker.process.pid, signal.SIGUSR1)

    async def on_receive_response(self):
        r, w = self._conn
        try:
            while not r.at_eof():
                result = await message_util.recv_response(r, timeout=None)
                logger.debug(f"Receiving: {result}")

                if result == (
                    message_util.Response.PONG,
                    message_util.RequestType.PING,
                ):
                    pass

                elif result == (
                    message_util.Response.RECONFIGURATION,
                    message_util.RequestType.UNDEFINED,
                ):
                    await self.on_receive_reconfiguration()

                elif result == (
                    message_util.Response.FORWARD_RANK0_PORT,
                    message_util.RequestType.UNDEFINED,
                ):
                    await self.on_receive_worker_port()
                elif result[0] == message_util.Response.SUCCESS:
                    response, request = result
                    if request not in self._response_callbacks:
                        logger.warning(f"Unexpected response: {request}")
                        continue

                    callback = self._response_callbacks.pop(request)
                    await callback()
                else:
                    logger.warning(f"Unexpected response: {result}")
                    continue

        except asyncio.IncompleteReadError:
            logger.info("Connection closed by master")
            return

    async def ping(self):
        loop = asyncio.get_running_loop()
        try:
            while loop.is_running():
                await asyncio.sleep(0.4)
                logger.debug("Sending ping")
                await self.send_request(message_util.RequestType.PING, None, None)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    args: OobleckAgentArguments = simple_parsing.parse(OobleckAgentArguments)
    agent = OobleckAgent(args)

    logger.info(f"Arguments: {args.to_dict()}")

    asyncio.run(agent.run())
