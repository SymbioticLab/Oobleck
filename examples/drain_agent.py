"""Request a safe graceful generation change through the control API."""

from __future__ import annotations

import asyncio

from oobleck.config import DrainConfig
from oobleck.elastic import request_drain


async def run(config: DrainConfig) -> None:
    generation = await request_drain(config.master_host, config.master_port, config.node_id)
    print(
        f"master accepted drain request for {config.node_id} at generation "
        f"{generation}; the active incarnation will publish the change"
    )


def main() -> None:
    import tyro

    asyncio.run(run(tyro.cli(DrainConfig)))


if __name__ == "__main__":
    main()
