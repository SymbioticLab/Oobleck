# Oobleck refactor examples

These role-specific examples use only the public plan/context, control-plane,
and DataLoader APIs. Install Oobleck with its exact reviewed Cornstarch commit,
including the `datasets` dependency. Profile and template fingerprints must
match the model, dtype, fixed per-node TP width, hardware, and Cornstarch
version used by the workers.

## One-node deployment

From the repository root:

```bash
python examples/run_local.py
```

This is a real subprocess smoke deployment: it starts a TCP membership master,
one CPU node agent, a Unix-domain worker relay, and one training worker. The
worker uses `FakeTextDataset` through a standard `torch.utils.data.DataLoader`,
prepares generation 1, waits for the CPU `generation_active` barrier, commits
one optimizer step, and drains the node into generation 2. CUDA is selected
when available; CPU remains a useful API smoke path.

The standalone trainer defaults to a tiny native PyTorch model so that the
control and data paths can be checked without downloading a checkpoint. To
exercise the public Cornstarch model and execution-plan APIs with the bundled
synthetic Llama configuration, use:

```bash
python examples/pretrain_llm.py --model-backend cornstarch
```

That backend requires Oobleck's exact pinned `refactor-oobleck` dependency.
Pass `--device cpu` for an API-only smoke run on a machine without CUDA.

## Dataset selection

`examples/pretrain_llm.py` defaults to deterministic synthetic tokens. Pass
`--dataset-name`, optional `--dataset-config`, and explicit `--split` arguments
to use `load_dataset(name, configuration, split="train")`. The returned
map-style `datasets.Dataset` is passed directly to `DataLoader`; no
`.with_format("torch")` call is required because the collator owns conversion.

A `DatasetDict` must be narrowed to a split before it reaches Oobleck. Streaming
and arbitrary iterable datasets are rejected because they lack the durable,
indexed cursor required for exact replay. Downloads use the normal Hugging Face
cache. Pre-populate it and set `HF_DATASETS_OFFLINE=1` for offline runs.
Tokenization and random augmentation must be deterministic from Oobleck logical
sample seeds or materialized ahead of time.

## Profile and template generation

The profile command can consume an existing versioned profile or call a Python
measurement factory. This repository includes a small executable factory:

```bash
python -m oobleck.cli profile-command-config \
  --output /tmp/tiny-templates.json \
  --model tiny-language \
  --measurement-factory examples.profile_tiny:create_workload \
  --dtype float32 \
  --microbatch-size 2 \
  --resource-counts 1 2 \
  --cornstarch-version CORNSTARCH_COMMIT
```

This writes `/tmp/tiny-templates.profile.json` and
`/tmp/tiny-templates.json`. A production factory returns
`ProfilingWorkload(layers, input_factory, loss_factory)` with materialized
Cornstarch LLM layers and representative per-layer inputs. The profiler
measures synchronized forward/backward time and persistent plus peak CUDA
memory. Cached profiles with an incompatible fingerprint are rejected with a
regeneration error.

## Manual multi-node launch

Start the master, then one agent per stable node. Each agent starts one worker
per listed GPU and does not invoke `torchrun`:

```bash
python examples/run_master.py --host 0.0.0.0 --port 29600

python examples/run_agent.py \
  --node-id node-a --master-host MASTER --master-port 29600 \
  --gpu-ids 0 1 2 3 --addresses 10.0.0.11 \
  --local-worker-socket /tmp/oobleck-node-a.sock \
  --worker-script examples/pretrain_llm.py

python examples/run_agent.py \
  --node-id node-b --master-host MASTER --master-port 29600 \
  --gpu-ids 0 1 2 3 --addresses 10.0.0.12 \
  --local-worker-socket /tmp/oobleck-node-b.sock \
  --worker-script examples/pretrain_llm.py
```

Node IDs remain stable across replacements; each agent process creates a fresh
incarnation. Membership proposals carry the previous active plan, and every
prepared worker acknowledges the same complete checksummed target plan. A newer
proposal supersedes older preparation, all agents acknowledge readiness, and only
then does the master publish `generation_active`.

For bootstrap only, an optional hostfile may start the initial agents over SSH:

```text
# NODE_ID ADDRESS GPU[,GPU...]
node-a 10.0.0.11 0,1,2,3
node-b 10.0.0.12 0,1,2,3
```

```bash
python -m oobleck.cli training-launch-config \
  --training-script examples/pretrain_llm.py \
  --initial-hostfile hosts.txt \
  --agent-script examples/run_agent.py \
  --master-host 10.0.0.10 --master-port 29600 \
  --max-nodes 16 --tensor-parallel-size 4
```

The hostfile is not consulted for later joins. New or replacement agents simply
run the normal agent command and self-register. A pure new-ID join may compile
early, but the master withholds rendezvous until incumbents finish their current
step, report prepared, and block the next step.

Inspect the full checksummed membership snapshot with:

```bash
python -m oobleck.cli inspect-membership-config \
  --master-host MASTER --master-port 29600
```

Multi-node NCCL churn is optional validation that requires dedicated machines.
Follow `docs/chaos_acceptance.md` for the opt-in manifest runner, simultaneous and
cascading failure scenarios, long-running leak bounds, and machine-readable
results. The regular suite exercises multi-rank model and optimizer recovery on
the single CUDA GPU over Gloo.

## Drain, failure, replacement, join, and replay

A graceful drain is a complete generation boundary:

```bash
python examples/drain_agent.py \
  --node-id node-b --master-host MASTER --master-port 29600
```

The failure helper is only for disposable example jobs. It verifies both the
process command line and stable node ID and requires an exact confirmation:

```bash
python examples/fail_agent.py \
  --pid PID --expected-node-id node-b --confirmation FAIL:node-b
```

Invoke it for two validated agents in the same detection window to demonstrate
a simultaneous failure. A replacement uses the same stable node ID in a fresh
agent process; a join uses a new ID. Both paths create a new WORLD rather than
editing groups in place. Replacements, drains, failures, and mixed changes are
hard transitions that replay an interrupted batch. A pure join lets the current
step commit once under the old generation, blocks the next step, restores the
new worker from that committed state and sampler cursor, and resumes with the
next batch without replay.

Expected logs show the proposal generation and reason set, full WORLD teardown,
compiled ownership, all-worker `prepared` consensus, coordinator rendezvous
publication, transfer schedule hash and source/destination byte balance, final
worker readiness and `generation_active`. Hard-transition logs then show replay
of the uncommitted logical batch and exactly one commit; pure-join logs instead
show `attempts=1`, the old-generation cutover commit, and the next logical batch
under the expanded generation. A cascading failure should show the partial
recovery marked superseded before the newest snapshot is prepared.

## Cleanup and troubleshooting

Stop workers and agents before the master. Normal shutdown removes the Unix
socket; remove a stale `/tmp/oobleck-*.sock` only after confirming no matching
agent is running. The failure helper sends `SIGKILL`, so use it only in a
disposable deployment.

If a node cannot join or activate, check TCP reachability, unique stable IDs,
identical GPU counts/TP width, worker-script paths on every host, writable Unix
socket directories, profile/template/runtime fingerprints, stable dataset
length/fingerprint, and that no stale master owns the rendezvous port. A profile
cache mismatch requires regeneration; a `RecoveryUnavailable` error means the
surviving resources cannot satisfy the requested replica or batch constraints.
