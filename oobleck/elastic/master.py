from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

import oobleck.elastic.message_util as message_util


@dataclass
class Job:
    name: str
    node_ips: List[str]
    init_worker_num: str


@dataclass
class DistributionInfo:
    ips: List[str]
    ranks: List[List[int]]


class OobleckMasterDaemon:
    """
    Master daemon process that manages the Oobleck cluster.
    Currently only supports a single job with a single master node and
    implemented in Python using asyncio with a single thread for proof of concept.

    A master daemon is responsible for:
    1. Launching agent processes into agent nodes as requested from user.
    2. Managing connections between agents and the master.
    3. Checking liveness of all connected agents.
    4. Broadcasting a failure message to all live agents if happens.
    5. Serving request of distributed information from agents.
    """

    def __init__(self):
        self._server: asyncio.Server | None = None
        self._port: int | None = None
        self._job: Job | None = None

    @property
    def port(self) -> int:
        return self._port

    @classmethod
    async def create(cls, master_port: int = 0) -> OobleckMasterDaemon:
        daemon = OobleckMasterDaemon()
        server: asyncio.Server = await asyncio.start_server(
            daemon.on_connected, "127.0.0.1", master_port
        )
        daemon._server = server
        daemon._port = server.sockets[0].getsockname()[1]

        return daemon

    async def run(self) -> OobleckMasterDaemon:
        logging.info(f"Master daemon is running on port {self._port}...")
        async with self._server:
            await self._server.serve_forever()

    async def request_job_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        """
        Temporary request handler without maintaining the connection.
        Store job information and launch agents.
        """
        result: message_util.Response
        try:
            if self._job:
                logging.warning("Job already exists.")
                result = message_util.Response.FAILURE
            else:
                self._job = await message_util.recv(r, need_pickle=True)
                # TODO: Implement job launch.
                result = message_util.Response.SUCCESS
        except Exception as e:
            result = message_util.Response.FAILURE
        finally:
            await message_util.send_response(w, result)

    async def get_dist_info_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        """
        Temporary request handler without maintaining the connection.
        Return distributed information stored in job launch.
        """
        if not self._job:
            logging.warning("No job exists.")
            await message_util.send_response(w, message_util.Response.FAILURE)
            return

        await message_util.send_response(w, message_util.Response.SUCCESS, close=False)
        await message_util.send(w, self._job, close=True)

    async def register_agent_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        logging.info("Received agent registration request.")
        pass

    async def on_connected(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        """
        Callback function when any process (either agent or user) connects to the master.
        """
        try:
            request_type = await message_util.recv_request_type(r)
            loop = self._server.get_loop()

            if request_type == message_util.RequestType.LAUNCH_JOB:
                loop.create_task(self.request_job_handler(r, w))
            elif request_type == message_util.RequestType.GET_DIST_INFO:
                loop.create_task(self.get_dist_info_handler(r, w))
            elif request_type == message_util.RequestType.REGISTER_AGENT:
                loop.create_task(self.register_agent_handler(r, w))
            else:
                logging.warning(f"Unknown request type: {request_type}")
                w.close()
                await w.wait_closed()
        except asyncio.IncompleteReadError:
            logging.warning("Connection closed unexpectedly.")
            w.close()
            await w.wait_closed()


async def main():
    daemon = await OobleckMasterDaemon.create()
    await daemon.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
