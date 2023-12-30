import multiprocessing
from concurrent import futures
from pathlib import Path

import grpc
import pytest
from oobleck.elastic import master_service_pb2_grpc
from oobleck.elastic.run import HostInfo, MasterArgs, MasterService

fake_host_info = [
    HostInfo("127.0.0.1", 2, 1234),
    HostInfo("127.0.0.2", 2, 1234),
    HostInfo("127.0.0.3", 2, 1234),
]


@pytest.fixture()
def server(tmp_path: Path):
    fake_master_args = MasterArgs(
        hostfile=Path(tmp_path / "hostfile"),
        code_path=Path(tmp_path / "testcode.py"),
        output_dir=tmp_path,
    )

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
