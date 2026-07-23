from __future__ import annotations

import asyncio

from oobleck.elastic import (
    LocalWorkerClient,
    MasterControlService,
    NodeAgentClient,
    inspect_membership,
    inspect_status,
    request_drain,
)

from oobleck.types import OobleckExecutionPlan, PipelineInstance, PipelineTemplate


def _test_plan(generation: int, instance_id: str = "shared") -> OobleckExecutionPlan:
    template = PipelineTemplate("control-test", ((0, 1), (1, 2)), 1, 0.0, 0.0)
    instance = PipelineInstance(instance_id, template, ("node-a", "node-b"), ((0,), (1,)), 0)
    return OobleckExecutionPlan(
        generation,
        (instance,),
        (("node-a", (0,)), ("node-b", (1,))),
        generation - 1,
        "compat",
    )


def test_master_accumulates_removals_and_additions_until_activation():
    from oobleck.elastic import MembershipSnapshot, NodeIdentity

    service = MasterControlService()
    a = NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",))
    b = NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",))
    c = NodeIdentity("node-c", "c1", ("127.0.0.1",), ("0",))

    service._begin_generation(MembershipSnapshot(1, (a, b), (), (a, b)))
    service._clear_accumulated_operations()

    failed = service._begin_generation(MembershipSnapshot(2, (a,), (b,), ()))
    assert failed.removed_nodes == (b,)
    assert failed.added_nodes == ()

    expanded = service._begin_generation(MembershipSnapshot(3, (a, c), (), (c,)))
    assert expanded.removed_nodes == (b,)
    assert expanded.added_nodes == (c,)

    cancelled = service._begin_generation(MembershipSnapshot(4, (a,), (c,), ()))
    assert cancelled.nodes == (a,)
    assert cancelled.removed_nodes == (b, c)
    assert cancelled.added_nodes == (c,)


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


def test_local_workers_must_agree_on_plan_and_compatibility(tmp_path):
    import pytest

    from oobleck.elastic import LocalWorkerRelay, MessageEnvelope, ProtocolError

    relay = LocalWorkerRelay(tmp_path / "relay.sock", "node-a", expected_workers=2)
    relay._latest_membership = MessageEnvelope(
        "membership",
        "master",
        "master",
        1,
        7,
        {
            "nodes": [],
            "removed_nodes": [],
            "added_nodes": [],
            "detection_seconds": 0.0,
            "snapshot_hash": "membership",
            "previous_execution_plan": None,
        },
    )
    relay._workers = {"gpu-0": object(), "gpu-1": object()}
    relay._acknowledged = {
        "gpu-0": ("membership", "plan-a", "compat"),
        "gpu-1": ("membership", "plan-b", "compat"),
    }
    assert relay.generation_ready(7)
    with pytest.raises(ProtocolError, match="disagree"):
        relay.readiness_metadata(7)


def test_master_rejects_cross_agent_plan_disagreement():
    async def check():
        from oobleck.elastic import AsyncioTcpControlTransport, MessageEnvelope

        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.05)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = AsyncioTcpControlTransport()
        first = await transport.connect("127.0.0.1", port)
        second = await transport.connect("127.0.0.1", port)
        await first.send(
            MessageEnvelope(
                "register",
                "node-a",
                "a1",
                0,
                0,
                {"addresses": ["127.0.0.1"], "gpu_ids": ["0"]},
            )
        )
        assert (await first.receive()).generation == 1
        await second.send(
            MessageEnvelope(
                "register",
                "node-b",
                "b1",
                0,
                0,
                {"addresses": ["127.0.0.1"], "gpu_ids": ["0"]},
            )
        )
        first_proposal = await first.receive()
        second_proposal = await second.receive()
        assert first_proposal.generation == second_proposal.generation == 2
        snapshot_hash = first_proposal.payload["snapshot_hash"]
        first_plan = _test_plan(2, "plan-a")
        second_plan = _test_plan(2, "plan-b")
        await first.send(
            MessageEnvelope(
                "generation_prepared",
                "node-a",
                "a1",
                1,
                2,
                {
                    "snapshot_hash": snapshot_hash,
                    "plan_checksum": first_plan.plan_checksum,
                    "compatibility_digest": "compat",
                    "execution_plan": first_plan.to_dict(),
                },
            )
        )
        await second.send(
            MessageEnvelope(
                "generation_prepared",
                "node-b",
                "b1",
                1,
                2,
                {
                    "snapshot_hash": snapshot_hash,
                    "plan_checksum": second_plan.plan_checksum,
                    "compatibility_digest": "compat",
                    "execution_plan": second_plan.to_dict(),
                },
            )
        )
        superseding = await asyncio.wait_for(first.receive(), timeout=2)
        assert superseding.message_type == "membership"
        assert superseding.generation == 3
        assert [item["agent_id"] for item in superseding.payload["removed_nodes"]] == ["node-b"]
        assert [item["agent_id"] for item in superseding.payload["added_nodes"]] == [
            "node-a",
            "node-b",
        ]
        assert "reasons" not in superseding.payload
        assert service.active_generation != 2
        await first.close()
        await second.close()
        await service.close()

    asyncio.run(check())


def test_master_enforces_prepared_rendezvous_ready_active_order():
    async def check():
        from oobleck.elastic import AsyncioTcpControlTransport, MessageEnvelope

        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.05)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = AsyncioTcpControlTransport()
        first = await transport.connect("127.0.0.1", port)
        second = await transport.connect("127.0.0.1", port)
        await first.send(
            MessageEnvelope(
                "register",
                "node-a",
                "a1",
                0,
                0,
                {"addresses": ["127.0.0.1"], "gpu_ids": ["0"]},
            )
        )
        await first.receive()
        await second.send(
            MessageEnvelope(
                "register",
                "node-b",
                "b1",
                0,
                0,
                {"addresses": ["127.0.0.1"], "gpu_ids": ["0"]},
            )
        )
        first_membership = await first.receive()
        second_membership = await second.receive()
        assert first_membership.generation == second_membership.generation == 2
        snapshot_hash = first_membership.payload["snapshot_hash"]
        execution_plan = _test_plan(2)
        metadata = {
            "snapshot_hash": snapshot_hash,
            "plan_checksum": execution_plan.plan_checksum,
            "compatibility_digest": "compat",
        }
        prepared_metadata = {**metadata, "execution_plan": execution_plan.to_dict()}

        await first.send(
            MessageEnvelope("generation_prepared", "node-a", "a1", 1, 2, prepared_metadata)
        )
        await asyncio.sleep(0.02)
        assert service._rendezvous_metadata is None
        assert service.active_generation != 2

        await second.send(
            MessageEnvelope("generation_prepared", "node-b", "b1", 1, 2, prepared_metadata)
        )
        first_rendezvous = await first.receive()
        second_rendezvous = await second.receive()
        assert first_rendezvous.message_type == "generation_rendezvous"
        assert second_rendezvous == first_rendezvous
        assert service.active_generation != 2

        await first.send(MessageEnvelope("generation_ready", "node-a", "a1", 2, 2, metadata))
        await asyncio.sleep(0.02)
        assert service.active_generation != 2

        await second.send(MessageEnvelope("generation_ready", "node-b", "b1", 2, 2, metadata))
        first_active = await first.receive()
        second_active = await second.receive()
        assert first_active.message_type == "generation_active"
        assert second_active == first_active
        assert service.active_generation == 2

        await first.close()
        await second.close()
        await service.close()

    asyncio.run(check())


def test_status_reports_prepared_ready_and_active_generation():
    async def check():
        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.02)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = NodeAgentClient("node-a", ("0",), heartbeat_interval_s=0.01)
        await client.connect("127.0.0.1", port)
        task = asyncio.create_task(client.run())
        for _ in range(100):
            status = await inspect_status("127.0.0.1", port)
            if status.active:
                break
            await asyncio.sleep(0.01)
        assert status.generation == 1
        assert status.active_generation == 1
        assert status.prepared_agents == ("node-a",)
        assert status.ready_agents == ("node-a",)
        await client.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.close()

    asyncio.run(check())


def test_local_worker_consumes_failure_while_graceful_boundary_waits(tmp_path):
    from dataclasses import asdict
    import time
    import threading

    from oobleck.elastic import LocalWorkerClient, MembershipSnapshot, MessageEnvelope, NodeIdentity

    previous = _test_plan(1)
    addition = MembershipSnapshot(
        2,
        (
            NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
            NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
        ),
        (),
        (NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),),
        previous_execution_plan=previous,
    )
    failure = MembershipSnapshot(
        3,
        (NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),),
        (NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),),
        (NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),),
        previous_execution_plan=previous,
    )

    def message(snapshot):
        return MessageEnvelope(
            "membership",
            "master",
            "master",
            snapshot.generation,
            snapshot.generation,
            {
                "nodes": [asdict(node) for node in snapshot.nodes],
                "removed_nodes": [asdict(node) for node in snapshot.removed_nodes],
                "added_nodes": [asdict(node) for node in snapshot.added_nodes],
                "detection_seconds": snapshot.detection_seconds,
                "snapshot_hash": snapshot.snapshot_hash,
                "previous_execution_plan": previous.to_dict(),
            },
        )

    class Connection:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(message(failure))

        async def receive(self):
            return await self.queue.get()

    class Context:
        def __init__(self):
            self.generation = 1
            self.release = threading.Event()
            self.prepared_execution_plan = previous

        def apply_membership(self, snapshot):
            self.generation = snapshot.generation
            self.prepared_execution_plan = _test_plan(snapshot.generation)
            if snapshot.generation == 3:
                self.release.set()

        def wait_until_generation_preparable(self, generation):
            if generation == 2:
                assert self.release.wait(timeout=5)
            return generation == self.generation

        def prepare_generation(self):
            return None

    async def check():
        worker = LocalWorkerClient(tmp_path / "unused.sock", "node-a", "gpu-0")
        worker.connection = Connection()
        worker.generation = 2
        started = time.monotonic()
        snapshot, plan = await worker._prepare_context_responsively(Context(), addition, None)
        assert time.monotonic() - started < 2
        assert snapshot.generation == plan.generation == 3

    asyncio.run(check())


def test_membership_binds_the_previous_plan_only_after_activation():
    async def check():
        service = MasterControlService(lease_timeout_s=2, lease_check_interval_s=0.02)
        server = await service.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        first = NodeAgentClient("node-a", ("0",), heartbeat_interval_s=0.01)
        await first.connect("127.0.0.1", port)
        first_task = asyncio.create_task(first.run())
        for _ in range(200):
            if service.active_generation == 1:
                break
            await asyncio.sleep(0.01)
        assert service.active_execution_plan is not None

        second = NodeAgentClient("node-b", ("0",), heartbeat_interval_s=0.01)
        proposal = await second.connect("127.0.0.1", port)
        previous = proposal.payload["previous_execution_plan"]
        assert previous is not None
        assert previous["plan_checksum"] == service.active_execution_plan.plan_checksum

        await second.close()
        await first.close()
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
        await service.close()

    asyncio.run(check())
