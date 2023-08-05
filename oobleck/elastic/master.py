from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from typing import Any, Optional

import aiofiles
import asyncssh
import simple_parsing
from deepspeed.utils.logging import LoggerFactory

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import (
    DistributedJobConfiguration,
    OobleckAgentArguments,
    flatten_configurations,
)

logger = LoggerFactory.create_logger("oobleck_master")

max_num_nodes: int = 32


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
    async def create(cls, ip: str | None, port: int) -> OobleckMasterDaemon:
        daemon = OobleckMasterDaemon()
        server: asyncio.Server = await asyncio.start_server(
            daemon.on_connected, ip, port
        )
        daemon._server = server
        daemon._port = server.sockets[0].getsockname()[1]

        return daemon

    async def run_node_agent(
        self,
        index: int,
        job_config: DistributedJobConfiguration,
        log_path: Path,
    ):
        master_ip = socket.gethostbyname(socket.gethostname())
        node_ip = job_config.node_ips[index]

        async with asyncssh.connect(
            node_ip, job_config.node_port, username=job_config.username
        ) as conn:
            cmd = '/bin/bash -ic "conda run --no-capture-output -n oobleck '
            cmd += "python -m oobleck.elastic.agent "
            agent_args = OobleckAgentArguments(
                master_ip=master_ip,
                master_port=self._port,
                node_ips=job_config.node_ips,
                job_args=job_config.job_args,
                num_workers=4,
            )
            cmd += " ".join(
                [f"--{k}={v}" for k, v in flatten_configurations(agent_args).items()]
            )
            cmd += '"'
            logger.info(f"Launching an agent on {node_ip}: {cmd}")

            log_file_path = log_path / f"{node_ip}.out"
            async with aiofiles.open(
                log_file_path, "w"
            ) as log_file, conn.create_process(
                cmd,
                term_type="xterm",
            ) as process:
                while not process.is_closing():
                    output = await process.stdout.readline()
                    await log_file.write(output)

    async def request_job_handler(
        self,
        job: DistributedJobConfiguration,
        r: asyncio.StreamReader,
        w: asyncio.StreamWriter,
    ):
        """
        Temporary request handler without maintaining the connection.
        Store job information and launch agents.
        """
        result: message_util.Response
        try:
            if self._job:
                raise RuntimeError("Job already exists.")

            self._job = job
            current_time = time.localtime(time.time())
            current_time = time.strftime("%m-%d-%Y-%H-%M-%S", current_time)

            log_path = Path(
                f"/tmp/oobleck/logs/{current_time}-{self._job.job_args.model_name}"
            )
            log_path.mkdir(parents=True, exist_ok=False)

            loop = self._server.get_loop()
            for index in range(len(self._job.node_ips)):
                loop.create_task(
                    self.run_node_agent(
                        index,
                        self._job,
                        log_path,
                    )
                )

            result = message_util.Response.SUCCESS
        except Exception as e:
            logger.warning(e)
            result = message_util.Response.FAILURE
        finally:
            await message_util.send_response(
                w, message_util.RequestType.LAUNCH_JOB, result
            )

    async def forward_rank0_port_handler(
        self, port: int, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        logger.debug(f"Received rank0 port: {port}")

        for _, writer in self._agent_connections.values():
            await message_util.send_response(
                writer,
                message_util.RequestType.UNDEFINED,
                message_util.Response.FORWARD_RANK0_PORT,
                close=False,
            )
            await message_util.send(
                writer,
                port,
                need_pickle=True,
                close=False,
            )

    async def register_agent_handler(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        # TODO: find another unique identifier than IP address.
        client_ip = w.get_extra_info("peername")[0]

        if self._job is None or client_ip not in self._job.node_ips:
            logger.warning(f"Agent {client_ip} is not registered")
            return await message_util.send_response(
                w,
                message_util.RequestType.REGISTER_AGENT,
                message_util.Response.FAILURE,
                close=True,
            )

        if client_ip in self._agent_connections:
            logger.warning(f"Agent {client_ip} already registered")
            return await message_util.send_response(
                w,
                message_util.RequestType.REGISTER_AGENT,
                message_util.Response.FAILURE,
                close=True,
            )

        logger.info(f"Registering agent stream: {client_ip}")
        self._agent_connections[client_ip] = (r, w)

        self._server.get_loop().create_task(self.agent_handler(client_ip))

        # self._server.get_loop().create_task(self.on_agent_callback(agent))
        await message_util.send_response(
            w,
            message_util.RequestType.REGISTER_AGENT,
            message_util.Response.SUCCESS,
            close=False,
        )

    async def close_agent(self, agent_ip: str):
        self._agent_connections.pop(agent_ip)

        # Broadcast reconfiguration event
        for _, w in self._agent_connections.values():
            await message_util.send_response(
                w,
                message_util.RequestType.UNDEFINED,
                message_util.Response.RECONFIGURATION,
                close=False,
            )
            await message_util.send(w, agent_ip, need_pickle=True, close=False)

    async def pong(self, w: asyncio.StreamWriter):
        logger.info("Sending pong")
        await message_util.send_response(
            w,
            message_util.RequestType.PING,
            message_util.Response.PONG,
            close=False,
        )

    async def agent_handler(self, agent_ip: str):
        loop = self._server.get_loop()
        r, w = self._agent_connections[agent_ip]
        try:
            while True:
                request_type = await message_util.recv_request_type(r)

                if request_type == message_util.RequestType.PING:
                    await self.pong(w)
                elif request_type == message_util.RequestType.FORWARD_RANK0_PORT:
                    port: int = await message_util.recv(r, need_pickle=True)
                    loop.create_task(self.forward_rank0_port_handler(port, r, w))
                else:
                    logger.warning(f"Unknown request type: {request_type}")
                    continue
        except (asyncio.IncompleteReadError, ConnectionResetError):
            logger.warning(f"Agent {agent_ip} disconnected")
            await self.close_agent(agent_ip)
            if not self._agent_connections:
                logger.warning("No agent alive. Cancel job.")
                self._job = None

    async def on_connected(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        """
        Callback function when any process (either agent or user) connects to the master.
        """
        try:
            request_type = await message_util.recv_request_type(r)
            loop = self._server.get_loop()

            logger.info(f"Received request: {request_type}")

            if request_type == message_util.RequestType.LAUNCH_JOB:
                job: DistributedJobConfiguration = await message_util.recv(
                    r, need_pickle=True
                )
                loop.create_task(self.request_job_handler(job, r, w))
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


async def main(ip: str | None, port: int):
    daemon = await OobleckMasterDaemon.create(ip, port)
    logger.info(f"Master daemon is running on port {daemon._port}...")
    await daemon._server.serve_forever()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--ip", type=Optional[str], default=None)
    parser.add_argument("--port", type=int, default=0)

    args = parser.parse_args()

    asyncio.run(main(args.ip, args.port))
