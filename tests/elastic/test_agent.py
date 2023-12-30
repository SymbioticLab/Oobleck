import functools
import os
from multiprocessing.connection import Connection
from unittest.mock import patch

import grpc
from oobleck.elastic.agent import Agent
from oobleck.elastic.master_service_pb2_grpc import OobleckMasterStub
from oobleck.elastic.run import HostInfo, MasterArgs, MasterService


def worker_main_forward_master_port(
    agent_index: int, pipe: Connection, gpu_index: int, code: bytes
):
    if agent_index == 0:
        pipe.send(4321)

    # Receive distributed info
    dist_info = pipe.recv()
    assert isinstance(dist_info, list)
    assert all(isinstance(host_info, HostInfo) for host_info in dist_info)

    # Receive port info
    port = pipe.recv()
    pipe.send(port)


def test_agent_forward_master_port(server: tuple[MasterArgs, MasterService, int]):
    _, __, port = server
    channel = grpc.insecure_channel(f"localhost:{port}")
    agent0 = Agent(0, OobleckMasterStub(channel))
    agent1 = Agent(1, OobleckMasterStub(channel))

    with patch(
        "oobleck.elastic.agent.Worker.worker_main",
        new=functools.partial(worker_main_forward_master_port, 0),
    ):
        agent0.launch_workers()
    with patch(
        "oobleck.elastic.agent.Worker.worker_main",
        new=functools.partial(worker_main_forward_master_port, 1),
    ):
        agent1.launch_workers()

    for agent in [agent0, agent1]:
        for worker in agent.workers:
            worker.process.join()
            assert worker.pipe.recv() == 4321
