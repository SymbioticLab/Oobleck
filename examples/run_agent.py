"""Register one node, launch its GPU workers, and maintain their lease."""

from __future__ import annotations

import asyncio

from oobleck.config import AgentConfig
from oobleck.elastic import MessageEnvelope, run_agent_service


async def run(config: AgentConfig) -> None:
    async def membership(message: MessageEnvelope) -> None:
        nodes = [item["agent_id"] for item in message.payload["nodes"]]
        removed = [item["agent_id"] for item in message.payload["removed_nodes"]]
        added = [item["agent_id"] for item in message.payload["added_nodes"]]
        print(
            f"node {config.node_id}: generation={message.generation} "
            f"members={nodes} removed={removed} added={added}",
            flush=True,
        )

    async def active(message: MessageEnvelope) -> None:
        print(
            f"node {config.node_id}: generation={message.generation} active",
            flush=True,
        )

    await run_agent_service(
        config,
        on_membership=membership,
        on_generation_active=active,
    )


def main() -> None:
    import tyro

    asyncio.run(run(tyro.cli(AgentConfig)))


if __name__ == "__main__":
    main()
