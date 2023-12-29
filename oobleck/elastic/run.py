from __future__ import annotations

import pickle
import socket
import sys
from concurrent import futures
from dataclasses import dataclass
from multiprocessing.context import SpawnContext
from pathlib import Path

import fabric
import grpc
from oobleck.elastic import master_service_pb2, master_service_pb2_grpc
import simple_parsing
from loguru import logger
from paramiko import SSHException

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

        return hosts


class MultiNodeAgentRunner:
    """
    A runner to execute multiple agents on multiple nodes.
    """

    def __init__(
        self,
        disconnect_condition: SpawnContext.Condition,
        hosts: list[HostInfo],
        master_service_port: int,
        output_dir: Path | None = None,
    ):
        self.disconnect_condition = disconnect_condition
        self.hosts = hosts
        self.master_service_port = master_service_port
        self.output_dir = output_dir

    @staticmethod
    def run_on_host(
        agent_index: int,
        disconnect_condition: SpawnContext.Condition,
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

        try:
            with fabric.Connection(host.ip, host.port) as conn:
                cmd = "python -m oobleck.elastic.agent "
                cmd += f"--master_ip {my_ip} --master_port {master_service_port} "
                cmd += f"--agent_index {agent_index}"

                with output.open(
                    "w"
                ) if output is not None else sys.stdout as out_stream:
                    conn.run(
                        cmd, hide=True, out_strea=out_stream, err_stream=out_stream
                    )
        except SSHException as e:
            # Notify conditional variable to notify agent disconnection
            # to all agents.
            logger.warning(f"SSH disconnected: {e}")
            disconnect_condition.notify_all()

    def run(self):
        """
        Spawn multiple processes to run agents on multiple hosts.
        Each process accesses a host via SSH and runs the agent.
        """
        with futures.ProcessPoolExecutor(mp_context=SpawnContext()) as pool:
            for agent_index, host in enumerate(self.hosts):
                output = (
                    self.outpur_dir / f"agent-{agent_index}.log"
                    if self.outpur_dir is not None
                    else None
                )
                pool.submit(
                    self.run_on_host,
                    agent_index,
                    self.disconnect_condition,
                    host,
                    self.master_service_port,
                    output,
                )

            pool.shutdown(wait=True)


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
        code_path: Path,
        hostinfo: list[HostInfo],
        disconnect_condition: SpawnContext.Condition,
    ):
        self.code = pickle.dumps(code_path.read_bytes())
        self.hostinfo = hostinfo
        self.disconnect_condition = disconnect_condition
        self.clients = []

    def GetDistInfo(self, request, context) -> master_service_pb2.DistInfo:
        return master_service_pb2.DistInfo(
            hosts=[
                master_service_pb2.HostInfo(
                    ip=host.ip, slots=host.slots, port=host.port
                )
                for host in self.hostinfo
            ]
        )

    def GetCode(self, request, context) -> master_service_pb2.CodeInfo:
        return master_service_pb2.CodeInfo(code=self.code)

    def ReceiveReconfigurationNotification(self, request, context):
        with self.condition:
            self.clients.append(context)
            self.disconnect_condition.wait()

        yield master_service_pb2.DistInfo(
            hosts=[
                master_service_pb2.HostInfo(
                    ip=host.ip, slots=host.slots, port=host.port
                )
                for host in self.hostinfo
            ]
        )


@dataclass
class MasterArgs:
    # Path to the hostfile
    hostfile: Path
    # Path to user training code
    code_path: Path
    # Port for master gRPC service
    master_service_port: int = 0
    # Directory to store agent logs
    output_dir: Path = Path("/tmp/oobleck")


def serve():
    args: MasterArgs = simple_parsing.parse(MasterArgs, dest="args")
    hostinfo = HostInfo.fetch_hostfile(args.hostfile)

    disconnect_condition = SpawnContext().Condition()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    service = MasterService(args.code_path, hostinfo, disconnect_condition)
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port(f"0.0.0.0:{args.master_service_port}")
    server.start()

    runner = MultiNodeAgentRunner(disconnect_condition, hostinfo, port, args.output_dir)
    runner.run()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
