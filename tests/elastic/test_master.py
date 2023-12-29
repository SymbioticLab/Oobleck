import threading
from concurrent import futures
from pathlib import Path

import grpc
from google.protobuf.empty_pb2 import Empty
import pytest

from oobleck.elastic import master_service_pb2, master_service_pb2_grpc
from oobleck.elastic.run import (
    HostInfo,
    MasterArgs,
    MasterService,
    MultiNodeAgentRunner,
)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    temp_dir = tmp_path_factory.mktemp("oobleck")

    fake_master_args = MasterArgs(
        hostfile=Path(temp_dir / "hostfile"),
        code_path=Path(temp_dir / "testcode.py"),
        output_dir=temp_dir,
    )

    fake_host_info = [
        HostInfo("127.0.0.1", 2, 1234),
        HostInfo("127.0.0.2", 2, 1234),
        HostInfo("127.0.0.3", 2, 1234),
    ]

    fake_master_args.hostfile.write_text(
        "\n".join(
            list(
                f"{host.ip} slots={host.slots} port={host.port}"
                for host in fake_host_info
            )
        )
    )

    fake_master_args.code_path.write_text("print('Hello, world!')")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    service = MasterService(
        fake_master_args.code_path, fake_host_info, threading.Condition()
    )
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port(f"0.0.0.0:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = master_service_pb2_grpc.OobleckMasterStub(channel)

    yield fake_master_args, stub
    server.stop(grace=None)


def test_get_dist_info(
    server: tuple[MasterArgs, master_service_pb2_grpc.OobleckMasterStub]
):
    fake_master_args, stub = server
    dist_info = stub.GetDistInfo(Empty())

    fake_dist_info = HostInfo.fetch_hostfile(fake_master_args.hostfile)
    assert len(dist_info.hosts) == len(fake_dist_info)
    for host, fake_host in zip(dist_info.hosts, fake_dist_info):
        assert host.ip == fake_host.ip
        assert host.slots == fake_host.slots
        assert host.port == fake_host.port


# @pytest.fixture(scope="module")
# def server() -> ThreadedServer:
#     t = ThreadedServer(MasterService, port=0)
#     t.protocol_config.update({"allow_public_attrs": True, "allow_setattr": True})
#     threading.Thread(target=t.start, daemon=True).start()

#     return t


# @pytest.fixture(autouse=True)
# def fake_service_variables(mocker: MockerFixture, tmp_path: Path):
#     fake_dist_args = DistArgs(
#         agent_ips=["127.0.0.1", "127.0.0.2", "127.0.0.3"],
#         world_size=6,
#         tensor_parallel_size=2,
#     )
#     mocker.patch("oobleck.run.dist_args", new=fake_dist_args)

#     fake_code_path = tmp_path / "fake_code.py"
#     fake_code_path.write_bytes("print('Hello, world!')".encode())

#     mocker.patch("oobleck.run.code_path", new=fake_code_path)
#     return fake_dist_args, fake_code_path


# def test_rpyc_functions(
#     server: ThreadedServer,
#     fake_service_variables: tuple[DistArgs, Path],
# ):
#     fake_dist_args, fake_code_path = fake_service_variables

#     conn = rpyc.connect("localhost", server.port)

#     dist_info = conn.root.get_dist_info()
#     code: bytes = brine.load(conn.root.get_code())

#     assert all(d == f for d, f in zip(dist_info.agent_ips, fake_dist_args.agent_ips))
#     assert dist_info.world_size == fake_dist_args.world_size
#     assert dist_info.backend == fake_dist_args.backend
#     assert dist_info.tensor_parallel_size == fake_dist_args.tensor_parallel_size
#     assert code == fake_code_path.read_bytes()


# def test_forward_port(server: ThreadedServer, mocker: MockerFixture):
#     connect_watcher = mocker.spy(server.service, "on_connect")
#     queue = Queue()

#     @rpyc.service
#     class FakeAgentService(rpyc.Service):
#         @rpyc.exposed
#         def receive_rank0_port(self, port: int):
#             queue.put(port)

#     conn = rpyc.connect("localhost", server.port, service=FakeAgentService)
#     # Wait until on_connect is called
#     while connect_watcher.call_count == 0:
#         pass

#     conn.root.forward_rank0_port(1234)
#     assert queue.get(timeout=10) == 1234


# def test_broadcast_reconfiguration(server: ThreadedServer, mocker: MockerFixture):
#     connect_watcher = mocker.spy(server.service, "on_connect")
#     queue = Queue()

#     @rpyc.service
#     class FakeAgentService(rpyc.Service):
#         @rpyc.exposed
#         def reconfigure(self, dist_args: DistArgs):
#             queue.put(dist_args)

#     conn1: Connection = rpyc.connect("localhost", server.port, service=FakeAgentService)
#     conn2: Connection = rpyc.connect("localhost", server.port, service=FakeAgentService)
#     conn3: Connection = rpyc.connect("localhost", server.port, service=FakeAgentService)
#     # Wait until on_connect is called
#     # while connect_watcher.call_count < 2:
#     #     pass
#     import time

#     time.sleep(5)

#     conn2.close()
#     conn1.serve(timeout=10)
#     conn3.serve(timeout=10)
#     new_dist_info: DistArgs = queue.get()
#     assert new_dist_info.agent_ips == ["127.0.0.2", "127.0.0.3"]
#     print(new_dist_info)
