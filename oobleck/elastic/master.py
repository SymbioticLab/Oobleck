from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import asdict
from pathlib import Path

import aiofiles
import asyncssh

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import (
    DistributedJobConfiguration,
    OobleckAgentArguments,
    OobleckArguments,
    flatten_configurations,
)

logger = logging.getLogger(__name__)

max_num_nodes: int = 32


# @dataclass
# class _AgentInfo:
#     """
#     OobleckAgent information.
#     A list of agent information is generated when a user requests a job to be launched.
#     First connected is set False, but will be set True when the agent connects to the master.
#     """

#     ip: str
#     ranks: list[int]
#     streams: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None


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
        self._job: DistributedJobConfiguration | None = None
        self._agent_connections: dict[
            str, tuple[asyncio.StreamReader, asyncio.StreamWriter]
        ] = {}

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
        logger.info(f"Master daemon is running on port {self._port}...")
        async with self._server:
            await self._server.serve_forever()

    async def run_node_agent(
        self,
        ip: str,
        port: int,
        username: str,
        job_config: OobleckArguments,
        log_path: Path,
    ):
        master_ip = socket.gethostbyname(socket.gethostname())
        async with asyncssh.connect(ip, port, username=username) as conn:
            cmd = '/bin/bash -ic "conda run -n oobleck python -m oobleck.elastic.agent '

            agent_args = OobleckAgentArguments(
                master_ip=master_ip,
                master_port=self._port,
                job_args=job_config,
                num_workers=4,
            )
            cmd += " ".join(
                [f"--{k}={v}" for k, v in flatten_configurations(agent_args).items()]
            )

            log_file_path = log_path / f"{ip}.out"
            async with conn.create_process(
                cmd, term_type="xterm"
            ) as process, aiofiles.open(log_file_path, "w") as log_file:
                output = await process.stdout.readline()
                await log_file.write(output)

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
                raise RuntimeError("Job already exists.")
            else:
                self._job: DistributedJobConfiguration = await message_util.recv(
                    r, need_pickle=True
                )

                current_time = time.localtime(time.time())
                current_time = time.strftime("%m-%d-%Y-%H-%M-%S", current_time)

                log_path = Path(
                    f"/tmp/oobleck/logs/{current_time}-{self._job.job_args.model_name}"
                )
                log_path.mkdir(parents=True, exist_ok=False)

                for node_ip in self._job.node_ips:
                    logger.info(f"Launching an agent on {node_ip}")
                    asyncio.create_task(
                        self.run_node_agent(
                            node_ip,
                            self._job.node_port,
                            self._job.username,
                            self._job.job_args,
                            log_path,
                        )
                    )

                result = message_util.Response.SUCCESS
        except Exception as e:
            logging.warning(e)
            result = message_util.Response.FAILURE
        finally:
            await message_util.send_response(
                w, message_util.RequestType.LAUNCH_JOB, result
            )

    async def get_dist_info_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        """
        Temporary request handler without maintaining the connection.
        Return distributed information stored in job launch.
        """
        if not self._job:
            logger.warning("No job exists.")
            await message_util.send_response(
                w,
                message_util.RequestType.GET_DIST_INFO,
                message_util.Response.FAILURE,
            )
            return

        # put client into waiting list
        self._nodes_to_rendezvous.add(w)

        logger.debug(
            f"Putting agent into waiting list "
            f"({len(self._nodes_to_rendezvous)} / {len(self._job.agent_info)})"
        )

        # if all clients are waiting, send dist info
        if len(self._nodes_to_rendezvous) == len(self._job.agent_info):
            logger.debug("Sending distributed information to agents")

            dist_info = message_util.DistributionInfo(
                agent_ips=[agent.ip for agent in self._job.agent_info],
                world_size=len(self._job.agent_info)
                * len(self._job.agent_info[0].ranks),
            )

            for w in self._nodes_to_rendezvous:
                await message_util.send_response(
                    w,
                    message_util.RequestType.GET_DIST_INFO,
                    message_util.Response.SUCCESS,
                    close=False,
                )
                await message_util.send(w, dist_info, need_pickle=True, close=False)
            self._nodes_to_rendezvous.clear()

    async def forward_rank0_port_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        port: int = await message_util.recv(r, need_pickle=True)
        logger.debug(f"Received rank0 port: {port}")
        for agent in self._job.agent_info:
            await message_util.send_response(
                agent.streams[1],
                message_util.RequestType.UNDEFINED,
                message_util.Response.FORWARD_RANK0_PORT,
                close=False,
            )
            await message_util.send(
                agent.streams[1],
                port,
                need_pickle=True,
                close=False,
            )

    async def register_agent_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        # TODO: find another unique identifier than IP address.
        client_ip = w.get_extra_info("peername")[0]

        agent: _AgentInfo = next(
            (agent for agent in self._job.agent_info if agent.ip == client_ip), None
        )

        if agent is None:
            logger.warning(f"Unknown agent: {client_ip}")
            await message_util.send_response(
                w,
                message_util.RequestType.REGISTER_AGENT,
                message_util.Response.FAILURE,
                close=True,
            )
            return

        # TODO: register callback on agent reader disconnection
        agent.streams = (r, w)
        logger.info("Registering agent stream")

        self._server.get_loop().create_task(self.agent_handler(agent))

        # self._server.get_loop().create_task(self.on_agent_callback(agent))
        await message_util.send_response(
            w,
            message_util.RequestType.REGISTER_AGENT,
            message_util.Response.SUCCESS,
            close=False,
        )

    async def close_agent(self, agent: _AgentInfo):
        _, w = agent.streams
        w.close()
        await w.wait_closed()
        self._job.agent_info.remove(agent)

        # Broadcast reconfiguration event
        for agent in self._job.agent_info:
            if agent.streams:
                await message_util.send_response(
                    agent.streams[1],
                    message_util.RequestType.UNDEFINED,
                    message_util.Response.RECONFIGURATION,
                    close=False,
                )

    async def pong(self, w: asyncio.StreamWriter):
        try:
            agent: _AgentInfo = next(
                agent
                for agent in self._job.agent_info
                if agent.streams and agent.streams[1] == w
            )
            logger.info("Sending pong")
            await message_util.send_response(
                agent.streams[1],
                message_util.RequestType.PING,
                message_util.Response.PONG,
                close=False,
            )
        except (AttributeError, StopIteration) as e:
            logger.warning(f"Unknown agent: {w.get_extra_info('peername')[0]}")
            w.close()
            await w.wait_closed()

    async def agent_handler(self, agent: _AgentInfo):
        r, w = agent.streams
        loop = self._server.get_loop()
        try:
            while True:
                request_type = await message_util.recv_request_type(r)

                if request_type == message_util.RequestType.PING:
                    await self.pong(w)
                elif request_type == message_util.RequestType.GET_DIST_INFO:
                    await self.get_dist_info_handler(r, w)
                elif request_type == message_util.RequestType.FORWARD_RANK0_PORT:
                    await self.forward_rank0_port_handler(r, w)
                else:
                    logger.warning(f"Unknown request type: {request_type}")
                    continue
        except asyncio.IncompleteReadError:
            logger.warning("Agent disconnected")
            await self.close_agent(agent)

    async def on_connected(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        """
        Callback function when any process (either agent or user) connects to the master.
        """
        try:
            request_type = await message_util.recv_request_type(r)
            loop = self._server.get_loop()

            logger.info(f"Received request: {request_type}")

            if request_type == message_util.RequestType.LAUNCH_JOB:
                loop.create_task(self.request_job_handler(r, w))
            elif request_type == message_util.RequestType.REGISTER_AGENT:
                loop.create_task(self.register_agent_handler(r, w))
            else:
                logger.warning(f"Unknown request type: {request_type}")
                w.close()
                await w.wait_closed()
        except asyncio.IncompleteReadError:
            logger.warning("Connection closed unexpectedly.")
            w.close()
            await w.wait_closed()


async def main():
    daemon = await OobleckMasterDaemon.create()
    await daemon.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
