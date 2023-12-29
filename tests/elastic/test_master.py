import threading
import multiprocessing
from concurrent import futures
from pathlib import Path

import grpc
from google.protobuf.empty_pb2 import Empty
import pytest
import pickle
from contextlib import redirect_stdout
from io import StringIO

from oobleck.elastic import master_service_pb2, master_service_pb2_grpc
from oobleck.elastic.run import (
    HostInfo,
    MasterArgs,
    MasterService,
    MultiNodeAgentRunner,
)


def get_stub(port: int) -> master_service_pb2_grpc.OobleckMasterStub:
    channel = grpc.insecure_channel(f"localhost:{port}")
    return master_service_pb2_grpc.OobleckMasterStub(channel)


@pytest.fixture()
def server(tmp_path: Path):
    fake_master_args = MasterArgs(
        hostfile=Path(tmp_path / "hostfile"),
        code_path=Path(tmp_path / "testcode.py"),
        output_dir=tmp_path,
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
        fake_master_args.code_path,
        fake_host_info,
        multiprocessing.get_context("spawn").Condition(),
    )
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port(f"0.0.0.0:0")
    server.start()

    yield fake_master_args, service, port
    server.stop(grace=None)


def test_get_dist_info(server: tuple[MasterArgs, MasterService, int]):
    fake_master_args, _, port = server
    stub = get_stub(port)
    dist_info = stub.GetDistInfo(Empty())

    fake_dist_info = HostInfo.fetch_hostfile(fake_master_args.hostfile)
    assert len(dist_info.hosts) == len(fake_dist_info)
    for host, fake_host in zip(dist_info.hosts, fake_dist_info):
        assert host.ip == fake_host.ip
        assert host.slots == fake_host.slots
        assert host.port == fake_host.port


def test_get_code(server: tuple[MasterArgs, MasterService, int]):
    fake_master_args, _, port = server
    stub = get_stub(port)
    code = pickle.loads(stub.GetCode(Empty()).code)
    assert code == fake_master_args.code_path.read_bytes()

    f = StringIO()
    with redirect_stdout(f):
        exec(code)
    assert f.getvalue() == "Hello, world!\n"


def test_receive_reconfiguration_notification(
    server: tuple[MasterArgs, MasterService, int]
):
    _, service, port = server

    from queue import Queue

    queue = Queue()

    def run_watcher(queue: Queue):
        stub = get_stub(port)
        dist_info = next(stub.WatchReconfigurationNotification(Empty()))
        queue.put(dist_info)

    threading.Thread(target=run_watcher, args=(queue,), daemon=True).start()

    # Wait until the thread is waiting
    while service.disconnect_condition._sleeping_count.get_value() == 0:
        pass

    assert service.disconnect_condition._sleeping_count.get_value() == 1

    # Manipulate dist info in service
    with service.disconnect_condition:
        service.hostinfo.pop(-1)
        service.disconnect_condition.notify_all()

    dist_info: master_service_pb2.DistInfo = queue.get(timeout=10)
    assert len(dist_info.hosts) == 2
    for host, fake_host in zip(dist_info.hosts, service.hostinfo[:-1]):
        assert host.ip == fake_host.ip
        assert host.slots == fake_host.slots
        assert host.port == fake_host.port
