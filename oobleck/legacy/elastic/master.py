from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from typing import Any

import aiofiles
import asyncssh
import simple_parsing
from deepspeed.utils.logging import LoggerFactory

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import OobleckArguments

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
        self._next_job_id: int = 0
        self._job_arguments: dict[int, OobleckArguments] = {}
        self._server: asyncio.Server | None = None
        self._port: int | None = None
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

    async def run_node_agents(
        self,
        args: OobleckArguments,
        node_index: int,
        agent_index: int,
        job_id: int,
        log_path: Path,
    ):
        node_ip = args.dist.node_ips[node_index]

        async with asyncssh.connect(
            node_ip, args.dist.node_port, username=args.dist.username
        ) as conn:
            cmd = '/bin/bash -ic "conda run --no-capture-output -n oobleck '
            cmd += "python -m oobleck.elastic.agent "
            cmd += f"--master_ip {args.dist.master_ip} --master_port {args.dist.master_port} "
            cmd += f'--job_id {job_id} --agent_index {agent_index}"'
            logger.info(f"Launching an agent on {node_ip}: {cmd}")

            log_file_path = log_path / f"{node_ip}.out"
            async with aiofiles.open(
                log_file_path, "w"
            ) as log_file, conn.create_process(
                cmd,
                term_type="xterm",
            ) as process:
                logger.info(
                    f"Agent {node_ip} output will be written at {log_file_path}."
                )
                async for data in process.stdout:
                    await log_file.write(data)
                    await log_file.flush()

    async def request_job_handler(
        self,
        args: OobleckArguments,
        r: asyncio.StreamReader,
        w: asyncio.StreamWriter,
    ):
        """
        Temporary request handler without maintaining the connection.
        Store job information and launch agents.
        """
        result: message_util.Response

        try:
            current_time = time.localtime(time.time())
            current_time = time.strftime("%m-%d-%Y-%H-%M-%S", current_time)

            log_path = Path(f"/tmp/oobleck/logs/{current_time}-{args.model.model_name}")
            log_path.mkdir(parents=True, exist_ok=False)

            self._job_arguments[self._next_job_id] = args

            loop = self._server.get_loop()
            for node_index in range(len(args.dist.node_ips)):
                for agent_index in range(args.dist.num_agents_per_node):
                    loop.create_task(
                        self.run_node_agents(
                            args,
                            node_index,
                            agent_index,
                            self._next_job_id,
                            log_path,
                        )
                    )

            result = message_util.Response.SUCCESS
        except Exception as e:
            logger.warning(e)
            result = message_util.Response.FAILURE
        finally:
            self._next_job_id += 1
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
        self, job_id: int, r: asyncio.StreamReader, w: asyncio.StreamWriter
    ):
        if job_id not in self._job_arguments:
            logger.warning(f"Agent {client_ip_port} sent a wrong job id")
            return await message_util.send_response(
                w,
                message_util.RequestType.REGISTER_AGENT,
                message_util.Response.FAILURE,
                close=True,
            )

        client_ip_port: tuple[str, int] = w.get_extra_info("peername")
        if client_ip_port in self._agent_connections:
            logger.warning(f"Agent {client_ip_port} already registered")
            return await message_util.send_response(
                w,
                message_util.RequestType.REGISTER_AGENT,
                message_util.Response.FAILURE,
                close=True,
            )

        logger.info(f"Registering agent stream: {client_ip_port}")
        self._agent_connections[client_ip_port] = (r, w)

        self._server.get_loop().create_task(self.agent_handler(client_ip_port))

        # self._server.get_loop().create_task(self.on_agent_callback(agent))
        await message_util.send_response(
            w,
            message_util.RequestType.REGISTER_AGENT,
            message_util.Response.SUCCESS,
            close=False,
        )
        await message_util.send(w, self._job_arguments[job_id])

    async def close_agent(self, agent_info: tuple[str, int]):
        self._agent_connections.pop(agent_info)

        # Broadcast reconfiguration event
        for _, w in self._agent_connections.values():
            await message_util.send_response(
                w,
                message_util.RequestType.UNDEFINED,
                message_util.Response.RECONFIGURATION,
                close=False,
            )
            await message_util.send(w, agent_info[0], need_pickle=True, close=False)

    async def pong(self, w: asyncio.StreamWriter):
        logger.info("Sending pong")
        await message_util.send_response(
            w,
            message_util.RequestType.PING,
            message_util.Response.PONG,
            close=False,
        )

    async def agent_handler(self, agent_info: tuple[str, int]):
        loop = self._server.get_loop()
        r, w = self._agent_connections[agent_info]
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
            logger.warning(f"Agent {agent_info} disconnected")
            await self.close_agent(agent_info)

    async def on_connected(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        """
        Callback function when any process (either agent or user) connects to the master.
        """
        try:
            request_type = await message_util.recv_request_type(r)
            loop = self._server.get_loop()

            logger.info(f"Received request: {request_type}")

            if request_type == message_util.RequestType.LAUNCH_JOB:
                args: OobleckArguments = await message_util.recv(r)
                loop.create_task(self.request_job_handler(args, r, w))
            elif request_type == message_util.RequestType.REGISTER_AGENT:
                job_id: int = await message_util.recv(r)
                loop.create_task(self.register_agent_handler(job_id, r, w))
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
    parser.add_argument("--ip", type=str, default="")
    parser.add_argument("--port", type=int, default=0)

    args = parser.parse_args()
    if not args.ip:
        args.ip = "127.0.0.1"

    asyncio.run(main(args.ip, args.port))
