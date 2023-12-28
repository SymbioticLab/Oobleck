import asyncio
import socket
import threading
from datetime import datetime
from pathlib import Path

import aiofiles
import asyncssh
import rpyc
from loguru import logger
from rpyc.core import brine
from rpyc.utils.server import ThreadedServer
from simple_parsing import ArgumentParser

from oobleck.arg_utils import DistArgs

"""
Oobleck master process code.
It uses rpyc to launch agents and communicate with them.
The master transfers the given serialized execution code
to agents, which will be executed in their worker processes.

After launching all worker processes, the master process
watches disconnection evernts from agents.
Once an agent is disconnected, the master process will
broadcast `reconfigure` message to all live agents.
"""
dist_args: DistArgs = None
code_path: Path = None
debug_mode: bool = False

log_path = Path("/tmp/oobleck/logs")


@rpyc.service
class MasterService(rpyc.Service):
    """A reactor class for the master process.

    This service maintains connections between the master process
    and agents; when an agent is lost, it broadcasts a reconfiguration
    message to all live agents.
    """

    agents: list[rpyc.Connection] = []

    @rpyc.exposed
    def get_dist_info(self) -> DistArgs:
        global dist_args
        return dist_args

    @rpyc.exposed
    def get_code(self) -> str:
        global code_path
        code: bytes = code_path.read_bytes()
        return brine.dump(code)

    @rpyc.exposed
    def forward_rank0_port(self, port: int):
        """The master rank sends its port to all agents."""
        pass

    def on_connected(self, conn: rpyc.Connection):
        self.agents.append(conn)

    def on_disconnect(self, conn: rpyc.Connection):
        self.agents.remove(conn)
        self.broadcast_reconfiguration()

    def broadcast_reconfiguration(self):
        try:
            for conn in self.agents:
                conn.root.reconfigure(dist_args)
        except Exception as e:
            print(f"Failed to broadcast reconfiguration: {e}")


async def run_agents(dist_args: DistArgs, master_port: int):
    """Run agents on all agent nodes.

    Running agents is done through SSH, and all outputs will be redirected to
    /tmp/oobleck/logs/{current_time}/{agent_ip}.out.
    """

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    master_ip = socket.gethostbyname(socket.gethostname())

    async def run_agent(agent_index: int, agent_ip: str):
        logger.debug(f"Connecting to agent {agent_ip}...")
        async with asyncssh.connect(agent_ip, username="insujang") as conn:
            cmd += "python -m oobleck.elastic.agent "
            cmd += f"--master_ip {master_ip} --master_port {master_port}"
            cmd += f"--agent index {agent_index}"

            log_file_path = log_path / current_time / f"{agent_ip}.out"
            logger.debug(f"Launching an agent on {agent_ip}: {cmd}")

            if debug_mode:
                async with conn.create_process(cmd, term_type="xterm") as process:
                    async for data in process.stdout:
                        print(data, end="")
            else:
                async with aiofiles.open(
                    log_file_path, "w"
                ) as log_file, conn.create_process(cmd, term_type="xterm") as process:
                    async for data in process.stdout:
                        await log_file.write(data)
                        await log_file.flush()

            exit_status = process.exit_status
            if exit_status is not None:
                logger.debug(f"Agent {agent_ip} exited with status {exit_status}.")

    coros = []
    for agent_index, agent_ip in enumerate(dist_args.agent_ips):
        coros.append(asyncio.create_task(run_agent(agent_index, agent_ip)))

    await asyncio.gather(*coros)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("code_path", type=Path, help="Path to the code to be executed.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print agent logs to stdout.",
    )
    parser.add_arguments(DistArgs, dest="dist_args")

    args = parser.parse_args()
    dist_args = args.dist_args
    code_path = args.code_path
    debug_mode = args.debug
    if debug_mode:
        logger.level("DEBUG")

    logger.debug(f"Dist args: {dist_args}")
    logger.debug(f"Code path: {code_path}")
    logger.debug(f"Debug mode: {debug_mode}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    t = ThreadedServer(MasterService)
    logger.info(f"Master is listening at {t.host}:{t.port}")
    threading.Thread(target=t.start, daemon=True).start()
    asyncio.run(run_agents(dist_args, t.port))
