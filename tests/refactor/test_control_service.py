from __future__ import annotations

import asyncio

from oobleck.elastic import (
    LocalWorkerClient,
    MasterControlService,
    NodeAgentClient,
    inspect_membership,
    request_drain,
)


def test_agents_join_and_concurrent_disconnects_publish_complete_snapshots():
    async def check():
        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.05)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        first = NodeAgentClient("node-a", ("0",), heartbeat_interval_s=0.02)
        second = NodeAgentClient("node-b", ("0",), heartbeat_interval_s=0.02)
        assert (await first.connect("127.0.0.1", port)).generation == 1
        first_task = asyncio.create_task(first.run())
        assert (await second.connect("127.0.0.1", port)).generation == 2
        second_task = asyncio.create_task(second.run())
        for _ in range(100):
            if first.generation == second.generation == 2:
                break
            await asyncio.sleep(0.01)
        assert first.generation == second.generation == 2
        for _ in range(100):
            if service.active_generation == 2:
                break
            await asyncio.sleep(0.01)
        assert service.active_generation == 2
        await asyncio.gather(first.connection.close(), second.connection.close())
        await asyncio.sleep(0.05)
        first_task.cancel()
        second_task.cancel()
        await asyncio.gather(first_task, second_task, return_exceptions=True)
        assert service.membership.agent_ids == ()
        assert service.membership.generation >= 3
        await service.close()

    asyncio.run(check())


def test_inspection_reconnect_and_graceful_drain_use_complete_snapshots():
    async def check():
        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.02)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = NodeAgentClient("node-a", ("0",), heartbeat_interval_s=0.01)
        await client.connect("127.0.0.1", port)
        first_incarnation = client.incarnation_id
        task = asyncio.create_task(client.run_reconnecting(retry_delay_s=0.01))

        inspected = await inspect_membership("127.0.0.1", port)
        assert [node.agent_id for node in inspected.nodes] == ["node-a"]
        await client.connection.close()
        for _ in range(200):
            if client.incarnation_id != first_incarnation and client.connection is not None:
                break
            await asyncio.sleep(0.01)
        assert client.incarnation_id != first_incarnation
        assert service.membership.agent_ids == ("node-a",)

        accepted_generation = await request_drain("127.0.0.1", port, "node-a")
        assert accepted_generation == client.generation
        for _ in range(200):
            if service.membership.agent_ids == ():
                break
            await asyncio.sleep(0.01)
        assert service.membership.agent_ids == ()
        await client.close()
        await asyncio.gather(task, return_exceptions=True)
        await service.close()

    asyncio.run(check())


def test_agent_relays_latest_membership_to_local_gpu_workers(tmp_path):
    async def check():
        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.02)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        socket_path = tmp_path / "agent.sock"
        first = NodeAgentClient(
            "node-a",
            ("0",),
            heartbeat_interval_s=0.01,
            local_worker_socket=socket_path,
        )
        await first.connect("127.0.0.1", port)
        first_task = asyncio.create_task(first.run())

        # A worker that starts after registration still receives generation 1.
        worker = LocalWorkerClient(socket_path, "node-a", "gpu-0")
        await worker.connect()
        assert (await worker.receive()).generation == 1

        second = NodeAgentClient("node-b", ("0",), heartbeat_interval_s=0.01)
        await second.connect("127.0.0.1", port)
        assert (await worker.receive()).generation == 2

        class Context:
            def __init__(self):
                self.generations = []
                self.active_generations = []

            def apply_membership(self, snapshot):
                self.generations.append(snapshot.generation)
                return True

            def mark_generation_active(self, generation):
                self.active_generations.append(generation)

        context = Context()
        worker_task = asyncio.create_task(worker.run_context(context))
        await second.close()
        for _ in range(200):
            if context.active_generations:
                break
            await asyncio.sleep(0.01)
        assert context.generations == [3]
        assert context.active_generations == [3]
        assert service.active_generation == 3

        await worker.close()
        await first.close()
        first_task.cancel()
        worker_task.cancel()
        await asyncio.gather(first_task, worker_task, return_exceptions=True)
        await service.close()
        assert not socket_path.exists()

    asyncio.run(check())
