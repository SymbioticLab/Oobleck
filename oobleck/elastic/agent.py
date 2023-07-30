import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import signal
from dataclasses import dataclass
from multiprocessing import connection
from typing import Any

import simple_parsing
from multiprocess.context import SpawnProcess

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import OobleckArguments
from oobleck.elastic.worker import worker_main

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    pipe: connection.Connection
    process: SpawnProcess


@dataclass
class OobleckAgentArguments:
    master_ip: str
    master_port: int
    oobleck_training_args: dict[str, Any]
    num_workers: int


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

    def __init__(self):
        self.conn_: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        self._workers: list[Worker] = []
        self.response_callbacks_: dict[message_util.RequestType, callable] = {}

    async def connect_to_master(self, master_ip: str, master_port: int):
        # TODO: add timeout in connection
        self.conn_ = await asyncio.wait_for(
            asyncio.open_connection(master_ip, master_port),
            timeout=message_util.TIMEOUT,
        )

    async def register_agent(self):
        await message_util.send_request_type(
            self.conn_[1], message_util.RequestType.REGISTER_AGENT
        )
        result, req = await message_util.recv_response(self.conn_[0])
        if (
            result is not message_util.Response.SUCCESS
            or req is not message_util.RequestType.REGISTER_AGENT
        ):
            raise RuntimeError("Failed to register agent")

        # When success, start pinging the master
        asyncio.create_task(self.ping())
        asyncio.create_task(self.on_receive_response())

    async def launch_workers(self, num_workers: int, args: OobleckArguments):
        context = mp.get_context("spawn")
        loop = asyncio.get_running_loop()
        for index in range(num_workers):
            # TODO: add all arguments. Arguments should be passed from the master
            # via command line arguments.
            pipe, child_pipe = context.Pipe()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
            worker = context.Process(
                target=worker_main,
                args=(index, num_workers, child_pipe, args),
                daemon=False,
            )
            worker.start()
            self._workers.append(Worker(pipe, worker))

        os.environ.pop("CUDA_VISIBLE_DEVICES")

        # TODO: detect worker failure and report it to the master.
        # For now only consider node failure, not worker failure.

    async def forward_worker_port(self, pipe: connection.Connection):
        _, w = self.conn_
        port: int = pipe.recv()
        await message_util.send_request_type(
            w, message_util.RequestType.FORWARD_RANK0_PORT
        )
        await message_util.send(w, port, need_pickle=True, drain=True, close=False)

    async def on_receive_worker_port(self):
        r, w = self.conn_
        port: int = await message_util.recv(r, need_pickle=True)
        for worker in self._workers:
            worker.pipe.send(port)

    async def send_request(
        self,
        request: message_util.RequestType,
        args: dict | None = None,
        callback: callable = None,
    ):
        if request in self.response_callbacks_:
            logger.warning(
                f"Already pending request for the same request type {request}"
            )
            return

        if request is not message_util.RequestType.PING:
            self.response_callbacks_[request] = callback
        await message_util.send_request_type(self.conn_[1], request)

        if args is not None:
            await message_util.send(
                self.conn_[1], args, need_pickle=True, drain=True, close=False
            )

    async def get_dist_info(self):
        await self.send_request(
            message_util.RequestType.GET_DIST_INFO, None, self.on_receive_dist_info
        )

    async def on_receive_dist_info(self):
        logger.debug("on_receive_dist_info")
        agent_info: message_util.DistributionInfo = await message_util.recv(
            self.conn_[0], need_pickle=True
        )

        for worker in self._workers:
            worker.pipe.send(agent_info)

    async def on_receive_reconfiguration(self):
        logger.debug("reconfiguration request received")
        # Send SIGUSR1 signal to workers
        for worker in self._workers:
            os.kill(worker.process.pid, signal.SIGUSR1)

    async def on_receive_response(self):
        r, w = self.conn_
        loop = asyncio.get_running_loop()
        try:
            while loop.is_running():
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
                    if request not in self.response_callbacks_:
                        logger.warning(f"Unexpected response: {request}")
                        continue

                    callback = self.response_callbacks_.pop(request)
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


async def main():
    args: OobleckAgentArguments = simple_parsing.parse(OobleckAgentArguments)

    agent = OobleckAgent()
    await agent.connect_to_master(args.master_ip, args.master_port)
    await agent.register_agent()
    agent.launch_workers(
        args.num_workers, OobleckArguments(**args.oobleck_training_args)
    )
    await agent.get_dist_info()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    asyncio.run(main())
