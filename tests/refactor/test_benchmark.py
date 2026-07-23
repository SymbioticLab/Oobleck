from benchmarks.recovery_schedule import run
from benchmarks.transaction_overhead import run as run_transaction_overhead


def test_recovery_schedule_benchmark_is_machine_readable_and_balanced():
    result = run(replicas=4, tensor_bytes=64, chunk_bytes=16)
    assert result["schema_version"] == 1
    assert result["balanced_max_source_bytes"] == 16
    assert result["legacy_first_source_max_bytes"] == 64
    assert result["max_source_load_reduction"] == 0.75
    assert result["transfer_count"] == 4
    assert result["round_count"] == 1
    assert result["maximum_round_source_bytes"] == 16
    assert result["maximum_round_destination_bytes"] == 64


def test_transaction_overhead_benchmark_replays_once_and_matches_reference():
    result = run_transaction_overhead(steady_steps=2)
    assert result["schema_version"] == 1
    assert result["baseline_step_seconds"] > 0
    assert result["oobleck_steady_step_seconds"] > 0
    assert result["recovery_step_seconds"] > 0
    assert result["recovery_attempts"] == 2
    assert result["committed_step"] == 3
    assert result["numerically_equivalent"] is True
    assert len(result["replayed_sample_indices"]) == 4
