from __future__ import annotations

import multiprocessing
import socket
import sys
from argparse import REMAINDER
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.synchronize import Condition
from pathlib import Path

import fabric
import grpc
from google.protobuf import empty_pb2
from loguru import logger
from oobleck.elastic import master_service_pb2, master_service_pb2_grpc
from paramiko import SSHException
from simple_parsing import ArgumentParser
from simple_parsing.helpers import field

"""
Oobleck master process code.
The master transfers the given serialized execution code
to agents, which will be executed in their worker processes.

After launching all agent processes, the master process
watches disconnection evernts from agents.
Once an agent is disconnected, the master process will
broadcast `reconfigure` message to all live agents.
"""


@dataclass
class LaunchArgs:
    # Path to the hostfile
    hostfile: Path
    # Port for master gRPC service
    master_service_port: int = 0
    # Directory to store agent logs
    output_dir: Path | None = None


@dataclass
class ScriptArgs:
    training_script: Path = field(positional=True)
    training_script_args: list[str] = field(positional=True, nargs=REMAINDER)


@dataclass
class HostInfo:
    ip: str
    slots: int
    port: int

    @staticmethod
    def fetch_hostfile(hostfile_path: Path) -> list[HostInfo]:
        """
        Parse the hostfile (MPI style) and return a list of HostInfo objects.

        A hostfile should look like:
        worker-0 slots=2 port=22
        worker-1 slots=2 port=1234
        worker-1 slots=2 port=1235

        The `slots` and `port` fields are optional.
        A hostname can be duplicated with different port, if agents are meant to be
        run on different Docker containers on the same host.
        """
        hosts: list[HostInfo] = []
        first_slots = None
        with hostfile_path.open("r") as f:
            for line in f.readlines():
                parts = line.split()
                # skip empty lines
                if not parts:
                    continue

                ip, slots, port = socket.gethostbyname(parts[0]), None, None
                for part in parts[1:]:
                    if part.startswith("slots="):
                        slots = int(part.split("=")[1])
                        if first_slots is None:
                            first_slots = slots
                        else:
                            assert (
                                slots == first_slots
                            ), "All hosts must have the same number of slots"
                    elif part.startswith("port="):
                        port = int(part.split("=")[1])

                if slots is None:
                    slots = 1
                if port is None:
                    port = 22

                hosts.append(HostInfo(ip, slots, port))

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
        output_dir: Path | None = None,
    ):
        self.disconnect_condition = disconnect_condition
        self.hosts = hosts
        self.master_service_port = master_service_port
        self.output_dir = output_dir

    @staticmethod
    def run_on_nodes(
        agent_index: int,
        disconnect_condition: Condition,
        host: HostInfo,
        master_service_port: int,
        output: Path | None = None,
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
                cmd += f"--agent_index {agent_index}"

                logger.debug(f"Connected to {host.ip}:{host.port}. Executing: {cmd}")

                out_stream = output.open("w") if output is not None else sys.stdout
                conn.run(cmd, hide=True, out_stream=out_stream, err_stream=out_stream)

                if output is not None:
                    out_stream.close()
        except SSHException as e:
            # Notify conditional variable to notify agent disconnection
            # to all agents.
            logger.warning(f"SSH disconnected: {e}")
            disconnect_condition.notify_all()

            print("Asdadas")

    def run(self):
        """
        Spawn multiple processes to run agents on multiple hosts.
        Each process accesses a host via SSH and runs the agent.
        """
        context = multiprocessing.get_context("spawn")
        for agent_index, host in enumerate(self.hosts):
            context.Process(
                target=self.run_on_nodes,
                args=(
                    agent_index,
                    self.disconnect_condition,
                    host,
                    self.master_service_port,
                    self.output_dir / f"agent-{agent_index}.log"
                    if self.output_dir is not None
                    else None,
                ),
            ).start()


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
        script_args: ScriptArgs,
        hostinfo: list[HostInfo],
        disconnect_condition: Condition,
    ):
        self.script_args = script_args
        self.hostinfo = hostinfo
        self.disconnect_condition = disconnect_condition
        self.clients = []
        self.master_port = 0

    def GetDistInfo(
        self,
        request: master_service_pb2.DistInfo,
        context: grpc.RpcContext,
    ) -> master_service_pb2.DistInfo:
        return master_service_pb2.DistInfo(
            hosts=[
                master_service_pb2.HostInfo(
                    ip=host.ip, slots=host.slots, port=host.port
                )
                for host in self.hostinfo
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

    def WatchReconfigurationNotification(
        self,
        request: empty_pb2.Empty,
        context: grpc.RpcContext,
    ):
        with self.disconnect_condition:
            self.clients.append(context)
            self.disconnect_condition.wait()

        if context.is_active():
            yield master_service_pb2.DistInfo(
                hosts=[
                    master_service_pb2.HostInfo(
                        ip=host.ip, slots=host.slots, port=host.port
                    )
                    for host in self.hostinfo
                ]
            )


def serve():
    parser = ArgumentParser()
    parser.add_arguments(LaunchArgs, dest="launch")

    # positional arguments
    parser.add_argument(
        "training_script",
        type=Path,
        help="Full path to the training script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )
    parser.add_argument("training_script_args", nargs=REMAINDER)

    args = parser.parse_args()
    launch_args: LaunchArgs = args.launch
    script_args = ScriptArgs(args.training_script, args.training_script_args)

    logger.info(f"Dist arguments: {launch_args}")
    logger.info(f"Script arguments: {script_args}")

    hostinfo = HostInfo.fetch_hostfile(launch_args.hostfile)

    server = grpc.server(ThreadPoolExecutor(max_workers=None))
    disconnect_condition = multiprocessing.get_context("spawn").Condition()
    service = MasterService(script_args, hostinfo, disconnect_condition)
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port(f"0.0.0.0:{launch_args.master_service_port}")
    server.start()
    logger.info(f"Running master service on port {port}")

    runner = MultiNodeAgentRunner(
        disconnect_condition, hostinfo, port, launch_args.output_dir
    )
    runner.run()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
