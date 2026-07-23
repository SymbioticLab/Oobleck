import multiprocessing
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from oobleck.elastic import master_service_pb2_grpc, run
from oobleck.elastic.run import (
    HostInfo,
    LaunchArguments,
    MasterService,
    ScriptArguments,
)

fake_host_info = [
    HostInfo("127.0.0.1", "0,1", 1234),
    HostInfo("127.0.0.2", "0,1", 1234),
    HostInfo("127.0.0.3", "0,1", 1234),
]


@pytest.fixture()
def server(
    tmp_path: Path,
) -> tuple[LaunchArguments, ScriptArguments, MasterService, int]:
    fake_launch_args = LaunchArguments(
        hostfile=Path(tmp_path / "hostfile"),
        tag="test-gpt2",
        base_dir=tmp_path,
    )

    fake_launch_args.hostfile.write_text(
        "\n".join(
            list(
                f"{host.ip} slots={len(host.devices.split(','))} devices={host.devices} port={host.port}"
                for host in fake_host_info
            )
        )
    )

    fake_script_args = ScriptArguments(
        training_script=Path(tmp_path / "testscript.py"),
        training_script_args=["--foo", "bar", "--baz", "qux"],
    )

    fake_script_args.training_script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--foo')\n"
        "parser.add_argument('--baz')\n"
        "args = parser.parse_args()\n"
        "print(f'Hello, {args.foo}, {args.baz}')\n"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    run.agent_list = [(host, None) for host in fake_host_info]
    service = MasterService(
        fake_script_args,
        multiprocessing.get_context("spawn").Condition(),
    )
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    port = server.add_insecure_port("0.0.0.0:0")
    server.start()

    yield fake_launch_args, fake_script_args, service, port
    server.stop(grace=None)
