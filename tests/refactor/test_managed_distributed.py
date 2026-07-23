from __future__ import annotations

import multiprocessing as mp
import socket
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
        ("initial-cohort",),
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

        replacement = MembershipSnapshot(
            5,
            (
                NodeIdentity("node-a", "a2", ("127.0.0.1",), ("0",)),
                NodeIdentity("node-b", "b2", ("127.0.0.1",), ("0",)),
            ),
            ("replacement-cohort",),
        )
        context.enable_control_plane_barrier()
        assert context.apply_membership(replacement)
        context.prepare_generation()
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
