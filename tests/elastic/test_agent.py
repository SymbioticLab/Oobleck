import pytest
import multiprocessing
from multiprocessing.connection import Connection
from unittest.mock import patch

import grpc
from oobleck.elastic.agent import Agent, Worker
from oobleck.elastic.master_service_pb2_grpc import OobleckMasterStub
from oobleck.elastic.run import HostInfo, MasterArgs, MasterService
from oobleck.engine.configuration_engine import ConfigurationEngine


@pytest.fixture(autouse=True)
def reset_configuration_engine():
    del ConfigurationEngine._instance
    ConfigurationEngine._instance = None


def worker_main_forward_master_port(
    pipe: Connection, agent_index: int, gpu_index: int, code: bytes
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
        new=worker_main_forward_master_port,
    ):
        agent0.launch_workers()
    with patch(
        "oobleck.elastic.agent.Worker.worker_main",
        new=worker_main_forward_master_port,
    ):
        agent1.launch_workers()

    for agent in [agent0, agent1]:
        for worker in agent.workers:
            worker.process.join()
            assert worker.pipe.recv() == 4321


@pytest.mark.parametrize("gpu_index", [0, 1, 2, 6])
def test_worker_main_init_configuration_engine(
    server: tuple[MasterArgs, MasterService, int],
    gpu_index: int,
):
    master_args, _, __ = server

    pipe, child_pipe = multiprocessing.Pipe()
    hosts = HostInfo.fetch_hostfile(master_args.hostfile)
    pipe.send(hosts)

    # This creates ConfigurationEngine instance.
    # Because Fake hostinfo has 2 losts per host,
    # it must raise IndexError when GPU index >= 2.
    if gpu_index >= 2:
        with pytest.raises(IndexError):
            Worker.worker_main(child_pipe, 0, gpu_index, b"")
        return

    Worker.worker_main(child_pipe, 0, 1, b"")

    assert ConfigurationEngine._instance is not None
    instance = ConfigurationEngine.get_instance()
    assert instance.agent_index == 0
    assert instance.local_rank == 1
    assert instance.dist_info == hosts
    assert instance.rank_map == {
        "127.0.0.1:1234": [0, 1],
        "127.0.0.2:1234": [2, 3],
        "127.0.0.3:1234": [4, 5],
    }
