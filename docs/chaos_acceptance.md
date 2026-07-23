# Optional multi-node NCCL chaos and churn validation

This is optional, opt-in destructive validation for dedicated machines. It is not
a completion or merge gate and is not run by the regular developer test
commands. Use disposable agent processes, the
exact pinned Cornstarch commit, identical per-node GPU counts, a shared metrics
directory, and a master address reachable from every worker.

## Start a long-running workload

Start the master and initial agents as described in `examples/README.md`. Put
`--worker-args` last because it forwards all remaining arguments to the training
worker. Each worker expands `{node_id}`, `{worker_id}`, and `{pid}` in the metric
path.

```bash
python examples/run_agent.py \
  --node-id node-a --master-host 10.0.0.10 --master-port 29600 \
  --gpu-ids 0 --addresses 10.0.0.11 \
  --local-worker-socket /tmp/oobleck-node-a.sock \
  --worker-script examples/pretrain_llm.py \
  --worker-args --steps 100000 --step-delay-s 0.01 \
  --metrics-output '/shared/oobleck-{worker_id}.jsonl'
```

CUDA selects NCCL automatically. Repeat this command with a stable identity and
address for every initial node. Before injecting failures, verify that status is
active:

```python
import asyncio
from oobleck.elastic import inspect_status

status = asyncio.run(inspect_status("10.0.0.10", 29600))
assert status.active
print(status.generation, [node.agent_id for node in status.snapshot.nodes])
```

## Configure and execute chaos

Copy `benchmarks/chaos_manifest.example.json`, replace every address and PID,
and adjust expected membership to the chosen template layout. Every command is
an explicit argv array; the runner never invokes a local shell. Commands must
return after performing or scheduling the action. The example uses `systemd-run
--user` for replacement agents so SSH returns while the agent continues.

The example covers:

- two agents killed concurrently across pipelines;
- a second failure injected after a newer membership proposal but before that
  proposal becomes active;
- replacement of a stable node identity and addition of a new identity;
- required observation of simple, borrow, and merge recovery strategies.

Run only after reviewing every command:

```bash
python benchmarks/chaos_acceptance.py \
  --manifest /path/to/chaos.json \
  --output benchmarks/results/multinode-nccl-chaos.json \
  --execute
```

Without `--execute`, the tool refuses to run. The cascading event also fails if
the intermediate generation becomes active before the second failure, ensuring
that the result really exercised recovery supersession rather than two ordinary
sequential failures.

## Long-running churn and leak checks

For a churn run, repeat drain/failure and replacement/join events in the
manifest for at least 25 generation changes. Keep the workload running across
the entire sequence. The verifier rejects:

- a worker generation moving backwards;
- skipped or duplicated committed-step metrics;
- a run with no replayed logical batch (`attempts > 1`);
- missing required simple/borrow/merge strategies;
- CUDA reserved-memory growth above
  `max_cuda_reserved_growth_bytes`;
- process-group-count growth above `max_process_group_growth`;
- a nonempty process-group universe on a requested clean close.

Abruptly killed workers cannot emit a clean-close record, so use
`require_clean_close: false` for failure campaigns. Use a separate graceful
finite-step campaign with `require_clean_close: true` to verify final WORLD and
subgroup retirement.

## Result schema

The output is versioned JSON containing:

- initial and final membership generations;
- per-event detection and total recovery duration;
- whether the event cascaded before activation;
- per-worker generation, step, replay, CUDA-memory, and process-group growth;
- observed reconfiguration strategies;
- Oobleck commit, Cornstarch/PyTorch/CUDA/NCCL/datasets versions, GPU model,
  runtime compatibility digests, and generation-plan checksums.

Worker JSONL records also contain the decomposed recovery timings and source
scheduling error. Preserve the manifest, worker JSONL files, master/agent logs,
and final result together for a reproducible paper-style run.
