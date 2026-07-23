"""Start the public asyncio membership master used by the examples."""

from __future__ import annotations

import asyncio

from oobleck.config import MasterServiceConfig
from oobleck.elastic import AsyncioTcpControlTransport, MasterControlService


async def run(config: MasterServiceConfig) -> None:
    service = MasterControlService(
        AsyncioTcpControlTransport(max_frame_bytes=config.max_frame_bytes),
        lease_timeout_s=config.lease_timeout_s,
        lease_check_interval_s=max(0.05, config.heartbeat_interval_s / 2),
        max_nodes=config.max_nodes,
    )
    server = await service.start(config.host, config.port)
    addresses = ", ".join(str(socket.getsockname()) for socket in server.sockets or ())
    print(f"Oobleck master listening on {addresses}")
    try:
        await server.serve_forever()
    finally:
        await service.close()


def main() -> None:
    import tyro

    asyncio.run(run(tyro.cli(MasterServiceConfig)))


if __name__ == "__main__":
    main()
