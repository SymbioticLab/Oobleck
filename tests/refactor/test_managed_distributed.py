from __future__ import annotations

import multiprocessing as mp
import socket
import threading
from typing import Any

import pytest
import torch


def _free_tcp_port() -> int:
    for base in range(30000, 60000):
        listeners = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
        try:
            listeners[0].bind(("127.0.0.1", base + 4))
            listeners[1].bind(("127.0.0.1", base + 5))
            return base
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()
    raise RuntimeError("could not reserve consecutive generation rendezvous ports")


def _managed_gloo_worker(rank: int, port: int, results: Any) -> None:
    import torch.distributed as dist

    from examples.pretrain_llm_base import build_training
    from oobleck.elastic import MembershipSnapshot, NodeIdentity

    torch.manual_seed(17)
    snapshot = MembershipSnapshot(
        4,
        (
            NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
            NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
        ),
        (),
        (
            NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
            NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
        ),
    )
    node_id = ("node-a", "node-b")[rank]
    model, context, loader = build_training(
        device="cpu",
        membership_snapshot=snapshot,
        local_node_id=node_id,
        local_tp_lane=0,
        max_nodes=2,
        rendezvous_port=port,
        distributed_backend="gloo",
    )
    try:
        assert dist.get_rank() == rank
        assert dist.get_world_size() == 2
        assert context.generation == 4
        assert context.owner_plan.rank_for_plan(context.execution_plan) == rank

        batch = next(iter(loader))
        step = context.step(
            batch,
            lambda microbatch: model(microbatch["input_ids"]),
            criterion=lambda logits, microbatch: torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), microbatch["labels"].flatten()
            ),
        )
        checksum = sum(parameter.detach().double().sum() for parameter in model.parameters())
        gathered = [torch.zeros_like(checksum) for _ in range(2)]
        dist.all_gather(gathered, checksum)
        torch.testing.assert_close(gathered[0], gathered[1])
        assert step.committed_step == 1

        restarted = MembershipSnapshot(
            5,
            (
                NodeIdentity("node-a", "a2", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b2", ("127.0.0.1",), ("0",)),
            ),
            (
                NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
            ),
            (
                NodeIdentity("node-a", "a2", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b2", ("127.0.0.1",), ("0",)),
            ),
        )
        context.enable_control_plane_barrier()
        assert context.apply_membership(restarted)
        context.prepare_generation()
        context.activate_generation()
        context.mark_generation_active(5)
        transition = context.recovery_history[-1]
        assert transition.generation == 5
        assert transition.detection_seconds == 0.0
        assert transition.configuration_planning_seconds >= 0.0
        assert transition.state_planning_seconds >= 0.0
        assert transition.state_transfer_seconds >= 0.0
        assert transition.straggler_round_seconds >= 0.0
        assert transition.source_scheduling_error_bytes == 0
        assert dist.get_rank() == rank
        assert dist.get_world_size() == 2

        next_batch = next(iter(loader))
        recovered_step = context.step(
            next_batch,
            lambda microbatch: model(microbatch["input_ids"]),
            criterion=lambda logits, microbatch: torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), microbatch["labels"].flatten()
            ),
        )
        results.put((rank, recovered_step.committed_step, recovered_step.generation))
    finally:
        context.close()
        assert not dist.is_initialized()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_membership_bootstraps_two_process_world_and_synchronized_step():
    port = _free_tcp_port()
    context = mp.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(target=_managed_gloo_worker, args=(rank, port, results))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("managed Gloo worker timed out")
        assert process.exitcode == 0
    observed = sorted(results.get(timeout=5) for _ in processes)
    assert observed == [(0, 2, 5), (1, 2, 5)]


def _graceful_addition_gloo_worker(
    rank: int,
    port: int,
    shared: Any,
    cohort_barrier: Any,
    cutover_barrier: Any,
    results: Any,
) -> None:
    import torch.distributed as dist

    from examples.pretrain_llm_base import (
        build_training,
        configure_training,
        prepare_training,
    )
    from oobleck.elastic import MembershipSnapshot, NodeIdentity
    from oobleck.types import OobleckExecutionPlan
    from tests.distributed.gloo_utils import (
        all_gather_gloo,
        all_to_all_gloo,
        all_to_all_single_gloo,
        batch_isend_irecv_gloo,
        reduce_scatter_gloo,
    )

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    original_all_reduce = dist.all_reduce

    def all_reduce_gloo(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if not tensor.is_cuda:
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)
        if async_op:
            raise ValueError("CUDA-over-Gloo all_reduce is synchronous in this test")
        cpu_tensor = tensor.cpu()
        original_all_reduce(cpu_tensor, op=op, group=group)
        tensor.copy_(cpu_tensor)
        return None

    dist.batch_isend_irecv = batch_isend_irecv_gloo
    dist.all_to_all = all_to_all_gloo
    dist.all_to_all_single = all_to_all_single_gloo
    dist.reduce_scatter = reduce_scatter_gloo
    dist.all_gather = all_gather_gloo
    dist.all_reduce = all_reduce_gloo

    initial = MembershipSnapshot(
        4,
        (
            NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
            NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
        ),
        (),
        (
            NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
            NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
        ),
    )
    node_id = ("node-a", "node-b", "node-c")[rank]
    context = None
    first_result = None
    first_indices = None
    if rank < 2:
        model, context, loader = build_training(
            device="cuda",
            membership_snapshot=initial,
            local_node_id=node_id,
            local_tp_lane=0,
            max_nodes=3,
            rendezvous_port=port,
            distributed_backend="gloo",
        )
        if rank == 0:
            shared["previous_plan"] = context.execution_plan.to_dict()
        cohort_barrier.wait()
        previous = OobleckExecutionPlan.from_dict(shared["previous_plan"])
        target = MembershipSnapshot(
            5,
            (
                NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-c", "c1", ("127.0.0.1",), ("0",)),
            ),
            (),
            (NodeIdentity("node-c", "c1", ("127.0.0.1",), ("0",)),),
            previous_execution_plan=previous,
        )
        context.enable_control_plane_barrier()
        batch = next(iter(loader))
        first_indices = tuple(batch.sample_indices)
        if rank == 0:
            shared["first_indices"] = first_indices
        entered = threading.Event()
        release = threading.Event()
        completed = []

        def criterion(logits, microbatch):
            if not entered.is_set():
                entered.set()
                assert release.wait(timeout=10)
            return torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), microbatch["labels"].to(logits.device).flatten()
            )

        trainer = threading.Thread(
            target=lambda: completed.append(
                context.step(
                    batch,
                    lambda microbatch: model(microbatch["input_ids"].to("cuda")),
                    criterion=criterion,
                )
            )
        )
        trainer.start()
        assert entered.wait(timeout=10)
        assert context.apply_membership(target)
        assert context._deferred_addition_plan is not None
        release.set()
        trainer.join(timeout=20)
        assert not trainer.is_alive()
        first_result = completed[0]
        assert first_result.attempts == 1
        assert first_result.generation == 4
        assert first_result.committed_step == 1
        assert context._pending_plan.generation == 5
        context.prepare_generation()
        cutover_barrier.wait()
        context.activate_generation()
        context.mark_generation_active(5)
        transition = context.recovery_history[-1]
        assert transition.removed_members == ()
        assert transition.added_members == (("node-c", "c1"),)
        assert transition.graceful_cutover
        assert transition.cutover_committed_step == 1
    else:
        cohort_barrier.wait()
        previous = OobleckExecutionPlan.from_dict(shared["previous_plan"])
        target = MembershipSnapshot(
            5,
            (
                NodeIdentity("node-a", "a1", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b1", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-c", "c1", ("127.0.0.1",), ("0",)),
            ),
            (),
            (NodeIdentity("node-c", "c1", ("127.0.0.1",), ("0",)),),
            previous_execution_plan=previous,
        )
        model, prepared, dataset = prepare_training(
            device="cuda",
            membership_snapshot=target,
            local_node_id=node_id,
            local_tp_lane=0,
            max_nodes=3,
            rendezvous_port=port,
            distributed_backend="gloo",
        )
        assert prepared.recover_from_survivors
        cutover_barrier.wait()
        context = prepared.activate()
        model, context, loader = configure_training(model, context, dataset, device="cuda")
        context.recover_from_survivors()

    try:
        assert context is not None
        assert dist.get_world_size() == 3
        assert context.generation == 5
        assert context._loaders[0].sampler.committed_cursor == 1
        next_batch = next(iter(loader))
        next_indices = tuple(next_batch.sample_indices)
        assert next_indices != tuple(shared["first_indices"])
        next_result = context.step(
            next_batch,
            lambda microbatch: model(microbatch["input_ids"].to("cuda")),
            criterion=lambda logits, microbatch: torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), microbatch["labels"].to(logits.device).flatten()
            ),
        )
        checksum = sum(parameter.detach().double().sum() for parameter in model.parameters())
        gathered = [torch.zeros_like(checksum) for _ in range(3)]
        dist.all_gather(gathered, checksum)
        for value in gathered[1:]:
            torch.testing.assert_close(value, gathered[0])
        results.put(
            (
                rank,
                1 if first_result is None else first_result.attempts,
                next_result.attempts,
                next_result.committed_step,
                next_result.generation,
                next_indices,
                context.execution_plan.plan_checksum,
            )
        )
    finally:
        if context is not None:
            context.close()
        assert not dist.is_initialized()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
def test_pure_addition_expands_gloo_world_without_replaying_committed_batch():
    port = _free_tcp_port()
    process_context = mp.get_context("spawn")
    manager = process_context.Manager()
    shared = manager.dict()
    cohort_barrier = process_context.Barrier(3)
    cutover_barrier = process_context.Barrier(3)
    results = process_context.Queue()
    processes = [
        process_context.Process(
            target=_graceful_addition_gloo_worker,
            args=(
                rank,
                port,
                shared,
                cohort_barrier,
                cutover_barrier,
                results,
            ),
        )
        for rank in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("graceful-addition Gloo worker timed out")
        assert process.exitcode == 0
    observed = sorted(results.get(timeout=5) for _ in processes)
    assert {(item[1], item[2], item[3], item[4]) for item in observed} == {(1, 1, 2, 5)}
    assert len({item[5] for item in observed}) == 1
    assert len({item[6] for item in observed}) == 1
    manager.shutdown()
