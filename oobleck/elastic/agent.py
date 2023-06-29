import asyncio
import logging
import multiprocessing as mp
import os
import signal
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import yaml
from multiprocess.context import SpawnProcess

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import TrainingArguments
from oobleck.elastic.worker import worker_main


@dataclass
class Worker:
    queue: mp.Queue
    process: SpawnProcess


@dataclass
class AgentArguments:
    master_ip: str
    master_port: int
    training_args: Path
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
        self.workers_: list[Worker] = []
        self.request_id_: int = 0
        self.on_response_callback_: dict[message_util.RequestType, callable] = {}

    async def connect_to_master(self, master_ip: str, master_port: int):
        # TODO: add timeout in connection
        self.conn_ = await asyncio.open_connection(master_ip, master_port)
        await self.register_agent()

    async def register_agent(self):
        await message_util.send_request_type(
            self.conn_[1], message_util.RequestType.REGISTER_AGENT
        )
        result = await message_util.recv_response(self.conn_[0])
        if result != message_util.Response.SUCCESS:
            raise RuntimeError("Failed to register agent")

        # When success, start pinging the master
        asyncio.create_task(self.ping())

    def launch_workers(self, num_workers: int, training_args: Path):
        context = mp.get_context("spawn")
        for _ in range(num_workers):
            # TODO: add all arguments. Arguments should be passed from the master
            # via command line arguments.
            queue = context.Queue()
            worker = context.Process(
                target=worker_main, args=(queue, training_args), daemon=False
            )
            worker.start()
            self.workers_.append(Worker(queue, worker))

            # TODO: detect worker failure and report it to the master.
            # For now only consider node failure, not worker failure.

    async def send_request(
        self,
        request: message_util.RequestType,
        args: dict | None = None,
        callback: callable | None = None,
    ):
        await message_util.send_request_type(self.conn_[1], request)

        if request in self.on_response_callback_:
            logging.warning("Already pending request for the same request type")
            return

        if request is not message_util.RequestType.PING:
            self.on_response_callback_[self.request_id_] = callback

            if args is not None:
                await message_util.send(
                    self.conn_[1], args, need_pickle=True, drain=True, close=False
                )

    async def ping(self):
        try:
            while True:
                await asyncio.sleep(0.4)
                await self.send_request(message_util.RequestType.PING, None, None)
        except asyncio.CancelledError:
            pass

    async def get_dist_info(self):
        await self.send_request(
            message_util.RequestType.GET_DIST_INFO, None, self.on_receive_dist_info
        )

    async def on_receive_dist_info(self, dist_info: dict):
        for worker in self.workers_:
            worker.queue.put(dist_info)

    async def on_receive_reconfiguration(self):
        # Send SIGUSR1 signal to workers
        for worker in self.workers_:
            os.kill(worker.process.pid, signal.SIGUSR1)

    async def on_receive_response(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                response, request = await message_util.recv_response(self.conn_[0])
                if response == message_util.Response.SUCCESS:
                    if request not in self.on_response_callback_:
                        logging.warning(f"Unexpected response: {request}")
                        continue

                    if request == message_util.RequestType.GET_DIST_INFO:
                        loop.create_task(self.on_receive_dist_info())
                    else:
                        pass
                elif response == message_util.Response.PONG:
                    pass
                elif response == message_util.Response.RECONFIGURATION:
                    loop.create_task(self.on_receive_reconfiguration())
        except asyncio.IncompleteReadError:
            logging.info("Connection closed by master")
            return


async def main(args: AgentArguments):
    agent = OobleckAgent()
    await agent.connect_to_master(args.master_ip, args.master_port)
    await agent.register_agent()
    agent.launch_workers(args.num_workers, args.training_args)
    await agent.get_dist_info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args: AgentArguments = simple_parsing.parse(AgentArguments)

    # check the given path is a valid yaml file
    try:
        TrainingArguments.load_yaml(args.training_args)
    except yaml.YAMLError as e:
        logging.error("Error parsing yaml file.")
        raise e

    asyncio.run(main(args))
