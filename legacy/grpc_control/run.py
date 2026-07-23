from __future__ import annotations

import multiprocessing
import socket
import sys
from concurrent import futures
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Condition
from pathlib import Path

import click
import fabric
import grpc
from google.protobuf import empty_pb2
from loguru import logger

from oobleck.elastic import master_service_pb2, master_service_pb2_grpc

"""
Oobleck master process code.
The master transfers the given serialized execution code
to agents, which will be executed in their worker processes.

After launching all agent processes, the master process
watches disconnection evernts from agents.
Once an agent is disconnected, the master process will
broadcast `reconfigure` message to all live agents.
"""

agent_list: list[tuple[HostInfo, Future]] = []


@dataclass
class LaunchArguments:
    # Path to the hostfile
    hostfile: Path
    # A tag to identify this run
    tag: str
    # Port for master gRPC service
    master_service_port: int
    # Oobleck root directory to store logs and profiles.
    base_dir: Path
    # Print agent's ssh outputs to stdout, instead of files
    debug: bool


@dataclass
class ScriptArguments:
    training_script: Path
    training_script_args: list[str]


class HostStatus(Enum):
    up = 0
    killed = 1
    terminating = 2


@dataclass
class HostInfo:
    ip: str
    devices: str
    port: int
    status: HostStatus = HostStatus.up

    def __eq__(self, other: HostInfo) -> bool:
        return (
            self.ip == other.ip
            and self.devices == other.devices
            and self.port == other.port
        )

    def __hash__(self) -> int:
        return hash((self.ip, self.devices, self.port))

    @staticmethod
    def fetch_hostfile(hostfile_path: Path) -> list[HostInfo]:
        """
        Parse the hostfile (MPI style) and return a list of HostInfo objects.

        A hostfile should look like:
        worker-0 slots=2 devices=0,1 port=22
        worker-0 slots=2 devices=2,3 port=22
        worker-1 slots=2 port=1234
        worker-1 slots=2 port=1235

        The `devices` and `port` fields are optional.

        You must specify the same number of slots to all agents.

        You can optionally specify the `devices` field to specify which GPUs
        should be used by the agent on the host.
        This will allow you to run multiple agents on the same host, with different GPUs.
        If not specified, `slots` number of GPUs starting from 0 will be used.

        If you use Docker containers to split GPUs on the same host,
        you can specify different port numbers for each container.
        """
        hosts: list[HostInfo] = []
        with hostfile_path.open("r") as f:
            for line in f.readlines():
                parts = line.split()
                # skip empty lines
                if not parts:
                    continue

                ip, slots, devices, port = (
                    socket.gethostbyname(parts[0]),
                    None,
                    None,
                    None,
                )
                first_slots = None
                for part in parts[1:]:
                    if part.startswith("slots="):
                        slots = int(part.split("=")[1])

                        if first_slots is None:
                            first_slots = slots

                        if first_slots != slots:
                            raise ValueError(
                                "All agents must have the same number of slots."
                            )
                    elif part.startswith("devices="):
                        devices = part.split("=")[1]
                    elif part.startswith("port="):
                        port = int(part.split("=")[1])

                if slots is None:
                    raise ValueError(
                        "The `slots` field must be specified for every agent."
                    )

                if devices is None:
                    devices = ",".join(str(i) for i in range(slots))
                if port is None:
                    port = 22

                host_info = HostInfo(ip, devices, port)
                if any(host == host_info for host in hosts):
                    raise ValueError(f"Duplicated host: {host_info}")

                hosts.append(host_info)

        logger.debug(f"Hosts: {hosts}")

        return hosts


class MultiNodeAgentRunner:
    """
    A runner to execute multiple agents on multiple nodes.
    """

    def __init__(
        self,
        disconnect_condition: Condition,
        hosts: list[HostInfo],
        master_service_port: int,
        tag: str,
        base_dir: Path | None = None,
    ):
        self.disconnect_condition = disconnect_condition
        self.hosts = hosts
        self.master_service_port = master_service_port
        self.tag = tag
        self.base_dir = base_dir

    @staticmethod
    def run_on_nodes(
        agent_index: int,
        host: HostInfo,
        master_service_port: int,
        tag: str,
        base_dir: Path,
        debug: bool,
    ):
        """
        Use fabric to run the agent on the given host.
        This function will block until the agent process is terminated.
        Therefore, it must be executed on a separate process.
        """
        my_ip = socket.gethostbyname(socket.gethostname())

        logger.info(
            f"Connecting to {host.ip}:{host.port} to instantiate an agent {agent_index}..."
        )

        try:
            with fabric.Connection(host.ip, port=host.port, connect_timeout=30) as conn:
                cmd = f"{sys.executable} -m oobleck.elastic.agent "
                cmd += f"--master_ip {my_ip} --master_port {master_service_port} "
                cmd += f"--agent_index {agent_index} "
                cmd += f"--tag {tag} --base_dir {str(base_dir)}"

                logger.debug(f"Connected to {host.ip}:{host.port}. Executing: {cmd}")

                if not debug:
                    out_stream = (base_dir / tag / f"agent{agent_index}.log").open("w")
                    logger.info(
                        f"Agent {agent_index} output will be saved to {(out_stream.name)}."
                    )
                else:
                    out_stream = sys.stderr

                conn.run(
                    cmd,
                    hide=True,
                    out_stream=out_stream,
                    err_stream=out_stream,
                )

        except Exception as e:
            logger.warning(f"[Agent {agent_index}] SSH disconnected: {e}")
            raise

        logger.info(f"[Agent {agent_index}] done. Exiting...")

    def run(self, debug: bool = False) -> list[SpawnProcess]:
        """
        Spawn multiple processes to run agents on multiple hosts.
        Each process accesses a host via SSH and runs the agent.
        """
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(self.hosts), mp_context=context
        ) as executor:
            global agent_list
            for agent_index, host in enumerate(self.hosts):
                future = executor.submit(
                    self.run_on_nodes,
                    agent_index,
                    host,
                    self.master_service_port,
                    self.tag,
                    self.base_dir,
                    debug,
                )
                agent_list.append((host, future))

            # Loop until all processes done
            while agent_list:
                not_done: list[Future] = [f for _, f in agent_list]
                done, not_done = futures.wait(
                    not_done, return_when=futures.FIRST_EXCEPTION
                )

                reconfiguration_needed = False
                for f in done:
                    if f.exception() is not None:
                        reconfiguration_needed = True

                    value = next(
                        (host, future) for host, future in agent_list if future == f
                    )
                    agent_list.remove(value)

                logger.debug("Remaining agents: {}", agent_list)

                if reconfiguration_needed:
                    with self.disconnect_condition:
                        self.disconnect_condition.notify_all()

            assert not agent_list and not not_done, "All agents should be done by now."


class MasterService(master_service_pb2_grpc.OobleckMasterServicer):
    """
    Master gRPC service.

    This service is used to transfer the serialized distributed info and
    user code to agents. It also broadcasts agent disconnect events to
    all agents. Broadcasting is done with server to client streaming gRPC
    and conditional variable; the cv is rung when an agent disconnects
    by MultiNodeAgentRunner after a ssh session to an agent is disconnected.
    """

    def __init__(
        self,
        script_args: ScriptArguments,
        disconnect_condition: Condition,
    ):
        self.script_args = script_args
        self.disconnect_condition = disconnect_condition
        self.master_port = 0

    def GetDistInfo(
        self,
        request: master_service_pb2.DistInfo,
        context: grpc.RpcContext,
    ) -> master_service_pb2.DistInfo:
        return master_service_pb2.DistInfo(
            hosts=[
                master_service_pb2.HostInfo(
                    ip=host.ip,
                    devices=host.devices,
                    port=host.port,
                    status=host.status.name,
                )
                for host, _ in agent_list
            ]
        )

    def GetCode(
        self,
        request: master_service_pb2.CodeInfo,
        context: grpc.RpcContext,
    ) -> master_service_pb2.CodeInfo:
        return master_service_pb2.CodeInfo(
            path=self.script_args.training_script.absolute().as_posix(),
            args=self.script_args.training_script_args,
        )

    def SetMasterRankPort(
        self,
        request: master_service_pb2.PortInfo,
        context: grpc.RpcContext,
    ) -> empty_pb2.Empty:
        self.master_port = request.port
        return empty_pb2.Empty()

    def GetMasterRankPort(
        self,
        request: empty_pb2.Empty,
        context: grpc.RpcContext,
    ) -> master_service_pb2.PortInfo:
        return master_service_pb2.PortInfo(port=self.master_port)

    def KillAgent(
        self, request: master_service_pb2.AgentInfo, context: grpc.RpcContext
    ) -> empty_pb2.Empty:
        agent_index = request.agent_index
        host, _ = agent_list[agent_index]

        host.status = HostStatus.terminating
        logger.info(f"Terminating agent {agent_index} on {host.ip}:{host.port}")

        with self.disconnect_condition:
            self.disconnect_condition.notify_all()

        return empty_pb2.Empty()

    def WatchReconfigurationNotification(
        self,
        request: empty_pb2.Empty,
        context: grpc.RpcContext,
    ):
        with self.disconnect_condition:
            self.disconnect_condition.wait()

        if context.is_active():
            yield master_service_pb2.DistInfo(
                hosts=[
                    master_service_pb2.HostInfo(
                        ip=host.ip,
                        devices=host.devices,
                        port=host.port,
                        status=host.status.name,
                    )
                    for host, _ in agent_list
                ]
            )


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--hostfile", type=Path, help="Path to the hostfile.")
@click.option("--tag", type=str, help="A tag to identify this run.")
@click.option(
    "--master_service_port", type=int, default=0, help="Port for master gRPC service."
)
@click.option(
    "--base_dir",
    type=Path,
    help="Oobleck root directory store logs and profiles.",
    default=Path("/tmp/oobleck"),
)
@click.option(
    "--debug",
    type=bool,
    is_flag=True,
    help="Print agent's ssh outputs to stderr, instead of files.",
    default=False,
)
@click.argument("training_script", type=Path)
@click.argument("training_script_args", nargs=-1, type=click.UNPROCESSED)
def serve(
    hostfile: Path,
    tag: str,
    master_service_port: int,
    base_dir: Path,
    debug: bool,
    training_script: Path,
    training_script_args: list[str],
):
    """
    training_script: Full path to the training script to be launched in parallel,
    followed by all the arguments for the training script.
    """
    launch_args = LaunchArguments(
        hostfile=hostfile,
        tag=tag,
        master_service_port=master_service_port,
        base_dir=base_dir,
        debug=debug,
    )
    script_args = ScriptArguments(
        training_script=training_script,
        training_script_args=training_script_args,
    )

    logger.info(f"Dist arguments: {launch_args}")
    logger.info(f"Script arguments: {script_args}")

    job_directory = launch_args.base_dir / launch_args.tag
    job_directory.mkdir(parents=True, exist_ok=True)
    assert job_directory.is_dir(), f"{str(job_directory)} is not a directory."

    hostinfo = HostInfo.fetch_hostfile(launch_args.hostfile)

    server = grpc.server(ThreadPoolExecutor(max_workers=None))
    disconnect_condition = multiprocessing.get_context("spawn").Condition()
    service = MasterService(script_args, disconnect_condition)
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port(f"0.0.0.0:{launch_args.master_service_port}")
    server.start()
    logger.info(f"Running master service on port {port}")

    runner = MultiNodeAgentRunner(
        disconnect_condition, hostinfo, port, launch_args.tag, launch_args.base_dir
    )
    runner.run(launch_args.debug)

    logger.info("Training is done. Stopping the master service.")
    server.stop(grace=None)


if __name__ == "__main__":
    serve()
