# Oobleck refactor implementation plan

## 1. Objective

Rebuild Oobleck on Cornstarch's `refactor` branch while preserving Oobleck's defining behavior:

- pre-generate pipeline templates and compose them into heterogeneous parallel configurations;
- choose model ownership before initializing `torch.distributed`;
- detect node failures through a dedicated CPU control plane rather than waiting for NCCL timeouts;
- completely destroy and recreate the distributed world for every membership generation;
- reconfigure with the simple, rank-borrowing, and pipeline-merge strategies from Section 5.1 of the Oobleck paper;
- transfer missing model and optimizer state efficiently, then replay the interrupted logical global batch;
- support initial deployment, one or multiple simultaneous abrupt failures, graceful drains, and joining or replacement nodes.

The first implementation supports LLM training with data, pipeline, and tensor parallelism. Tensor parallel size is fixed per node. Context parallelism, expert parallelism, multimodal models, and elastic changes to tensor-parallel width are deferred.

## 2. Branch and pull-request policy

### Oobleck

1. Create `refactor` from the current `develop` head.
2. Create a small feature branch for each subtask below.
3. Open each pull request against Oobleck `refactor`.
4. Run the subtask's focused tests and the accumulated refactor test suite before merging.
5. Approve and squash-merge passing Oobleck pull requests without waiting for additional approval.

### Cornstarch

1. Create `refactor-oobleck` from Cornstarch `refactor`.
2. Put only generally reusable runtime changes in this branch.
3. Open one Cornstarch pull request from `refactor-oobleck` to `refactor` and leave it open for upstream review; do not merge it automatically.
4. Pin Oobleck to the exact Cornstarch pull-request head commit rather than a moving branch.

The dependency boundary is important: Cornstarch should expose generic ways to compile and activate a local parallel partition, while Oobleck remains responsible for template planning, elasticity, membership, heterogeneous pipeline composition, and recovery policy.

## 3. Findings from Cornstarch `refactor`

Cornstarch `refactor` is not API-compatible with the Cornstarch version used by the current Oobleck. It now provides:

- meta-device Cornstarch model construction;
- `ParallelConfig`, `ParallelizationPlan`, and `ParallelContext`;
- a five-dimensional `DeviceMesh` for DP, PP, CP, TP, and EP;
- explicit pipeline schedules, including 1F1B;
- DTensor-based tensor parallelism.

The old Oobleck integration points no longer exist. In particular, the new branch has no old `PipelineTemplate`, heterogeneous plugin, ColossalAI placeholder machinery, or elastic reconfiguration API. Its current `ParallelizationPlan.materialize()` also assumes an already initialized distributed world and destructively slices the model's repeated `ModuleList` to the local pipeline stage.

Those assumptions conflict with Oobleck in three ways:

1. Oobleck must decide rank-local model ownership before WORLD exists.
2. Oobleck needs replicas with different pipeline depths and stage boundaries in the same job.
3. Oobleck must repeatedly retire and activate partitions while keeping a reusable model blueprint and stable logical state identities.

The refactor must therefore add a small set of generic lifecycle hooks to Cornstarch, then build Oobleck's heterogeneous elastic runtime above them.

## 4. Ownership boundary

### Cornstarch changes

Cornstarch owns generic, non-elastic primitives:

- process-group-independent compilation of rank-local ownership;
- explicit pipeline stage specifications with global layer ranges;
- a reusable meta-model blueprint;
- separate prepare and activate phases for a parallel partition;
- stable logical parameter and buffer identities;
- clean retirement of a `ParallelContext` and the process groups it owns;
- compatibility with Cornstarch's normal homogeneous materialization path.

### Oobleck changes

Oobleck owns research-system policy and orchestration:

- profiling and pipeline-template generation;
- heterogeneous template composition and batch allocation;
- simple, borrow, and merge reconfiguration;
- membership generations and node lifecycle;
- complete WORLD teardown and recreation;
- construction of heterogeneous pipeline meshes and schedules;
- synchronization groups across heterogeneous replicas;
- all-to-all state redistribution;
- replayable input and training-step commit semantics;
- failure-detection control plane and CLI.

This keeps Oobleck-specific policy out of Cornstarch while avoiding copies of Cornstarch internals in Oobleck.

## 5. Target public API

The new API should mirror Cornstarch's plan/context style rather than preserve the legacy `ExecutionEngine` surface.

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

model = from_hf_config(config, model_kind="language")
model.set_checkpoint_init(model_name_or_path=model_name)

plan = OobleckParallelizationPlan(
    OobleckConfig(
        global_batch_size=128,
        microbatch_size=2,
        fault_tolerance_threshold=1,
        max_nodes=16,
        seed=42,
    )
)
plan.parallelize(
    model,
    ParallelConfig(
        tensor_parallel_size=8,
    ),
)

context = plan.materialize("cuda", dtype=torch.bfloat16)

# Passing split= returns a map-style datasets.Dataset rather than DatasetDict.
dataset = load_dataset(dataset_name, dataset_config, split="train")
batch_sampler = context.create_batch_sampler(dataset, shuffle=True)
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
)
loader = context.prepare_dataloader(dataloader)

context.configure_optimization(
    optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-4),
    scheduler_factory=lambda optimizer: make_scheduler(optimizer),
)

for batch in loader:
    execution_plan, output = build_execution_plan(model)
    result = context.step(batch, execution_plan, output, criterion)
```

For the first release, the supplied Cornstarch `ParallelConfig` must have pipeline parallelism unset, DP/CP/EP equal to one, and a tensor-parallel size equal to the fixed GPUs per node. Oobleck chooses heterogeneous PP and effective DP from its templates.

Initial public types:

- `OobleckConfig`: training, profiling, resilience, membership, and control-plane settings;
- `OobleckParallelizationPlan`: creates or loads templates and prepares rank-local ownership;
- `OobleckParallelContext`: owns the active generation and exposes training/reconfiguration operations;
- `PipelineTemplate`: profiled stage partition, resource count, and predicted iteration time;
- `PipelineInstance`: a template assigned to concrete nodes/ranks in one generation;
- `OobleckExecutionPlan`: immutable generation configuration and communication topology;
- `OobleckBatchSampler`: standard PyTorch batch sampler that assigns logical global-batch indices to heterogeneous pipelines without advancing the committed cursor;
- `OobleckBatch`: replayable logical global batch with deterministic sample/microbatch identities;
- `RecoveryUnavailable`: explicit error when the remaining resources cannot form a valid fault-tolerant configuration.

## 6. Cornstarch upstream pull request

### 6.1 Split compile from activation

Refactor `ParallelizationPlan.materialize()` into two phases:

```python
compiled = plan.compile(
    world_size=world_size,
    rank=rank,
    stage_overrides=stage_specs,
)
context = compiled.activate(device="cuda", dtype=dtype, mesh=mesh)
```

`compile()` must not call `torch.distributed`, construct a `DeviceMesh`, or allocate real parameter storage. It computes only logical ownership and returns an immutable `CompiledParallelizationPlan`. `activate()` runs after Oobleck creates the new WORLD and subgroup topology; it attaches DTensor TP placements, materializes only local tensors, and creates the schedule.

Keep `materialize()` as a compatibility wrapper that performs both phases for existing Cornstarch users.

### 6.2 Add explicit stage specifications

Add a public `PipelineStageSpec` containing at least:

- pipeline and stage identifiers;
- global inclusive/exclusive layer range;
- owning global ranks and TP rank ordering;
- first/last-stage flags;
- optional tied-parameter ownership metadata.

Allow stage ranges to be uneven and supplied by an external planner. Normal Cornstarch planning should continue generating the same homogeneous ranges it does today.

### 6.3 Preserve the model blueprint

Replace destructive, one-shot slicing with a lifecycle that retains the root meta model and its full repeated-layer blueprint. Preparing a partition should create a local view without permanently deleting the other global layers from the blueprint. Repeated prepare/activate/retire cycles must preserve the root model object's identity so user-created execution plans and output references remain valid.

### 6.4 Stable logical state manifests

Expose a local manifest for parameters and persistent buffers. Each entry must include:

- a stable global logical key independent of current pipeline stage and rank;
- global shape, dtype, and state kind;
- current local shard shape and DTensor placements;
- global layer identity, when applicable;
- tied/shared state identity;
- the TP lane or shard coordinate needed to match sources and destinations.

Optimizer-state keys remain an Oobleck concern, but they will be derived from these parameter keys.

### 6.5 Explicit context cleanup

Add idempotent `ParallelContext.close()` behavior that releases Cornstarch-owned schedules, meshes, and subgroup references without assuming WORLD will remain alive. Oobleck will still perform the forceful global teardown.

### 6.6 Cornstarch acceptance tests

- compile ownership without initializing `torch.distributed`;
- compile explicit uneven stage ranges;
- activate and materialize only the local stage and TP shard;
- retain stable logical manifest keys across different stage assignments;
- preserve root model identity across repeated partition swaps;
- make `close()` idempotent;
- prove the existing homogeneous `materialize()` API and tests remain unchanged.

## 7. Oobleck architecture

Organize the rebuilt runtime into explicit layers:

```text
User training loop
  -> OobleckParallelContext
      -> Step transaction / replayable batch
      -> Active generation
          -> Heterogeneous pipeline runtimes
          -> Gradient synchronization topology
      -> Reconfiguration coordinator
          -> Membership generation
          -> Template composer (simple / borrow / merge)
          -> State redistribution plan
      -> Cornstarch compiled local partition
      -> Elastic control-plane client
```

The control plane chooses membership; the planner converts membership into a configuration; the runtime activates that configuration; the step transaction ensures failures never partially commit a logical global batch.

## 8. Pipeline profiling and template planning

Reuse the existing Rust planner for the first refactor, behind a narrow Python API, but define new serialized schemas that do not expose legacy Cornstarch or ColossalAI objects. Limit Rust changes in this effort to correctness, compatibility, tests, and binding maintenance; a structural planner refactor is explicitly deferred to `tasks/backlog.md`.

As part of bringing the planner forward, upgrade both `pyo3` and `pyo3-build-config` to at least `0.24.1`, regenerate `Cargo.lock`, adapt the bindings to the selected PyO3 API, and run the Rust and Python-extension tests. Do not resolve the current warning by pinning or allowing PyO3 `0.24.0`.

For each supported resource count, produce a `PipelineTemplate` with:

- ordered global layer ranges;
- per-stage nodes and fixed TP width;
- predicted forward, backward, and communication time;
- activation-memory and persistent-state requirements;
- maximum microbatch count and predicted iteration time;
- compatibility fingerprint containing model, dtype, TP, hardware, and Cornstarch version.

Template generation must enforce at least `fault_tolerance_threshold + 1` pipeline replicas when resources allow. Cache profiles using a versioned JSON/msgpack schema and reject stale or incompatible profiles with an actionable error.

The configuration solver chooses a multiset of templates and assigns the fixed global batch across them. Batch allocation should minimize predicted iteration time while ensuring every microbatch belongs to exactly one pipeline and gradients have the correct global weighting.

## 9. Initial heterogeneous runtime

Before WORLD initialization, every worker receives a signed or checksummed generation plan, determines its pipeline/stage/TP coordinate, asks Cornstarch to compile that rank-local ownership on the meta model, and records its expected state manifest.

After all agents acknowledge preparation:

1. The coordinator publishes the rendezvous parameters and ordered rank map.
2. Workers initialize WORLD for the new generation.
3. Oobleck creates per-pipeline PP/TP meshes and Cornstarch activates the local partition.
4. Oobleck creates cross-pipeline synchronization groups by logical layer and TP lane.
5. Workers load initial checkpoint state or receive migrated state.
6. The generation becomes active only after a final all-worker readiness barrier through the CPU control plane.

Pipelines may have different stage counts and layer boundaries. For each logical parameter, Oobleck synchronizes gradients among the replicas that own that parameter, matching the same TP shard/placement. Gradient scaling must use the number of samples contributed by each heterogeneous pipeline so the update equals the configured global batch.

Use Cornstarch's pipeline schedule within each pipeline, initially 1F1B. Validate that uneven stage layouts, differing microbatch counts, tied embeddings, and loss placement are handled explicitly.

## 10. Failure detection and membership generations

Retain a dedicated CPU agent per node and a master membership service, but do not require gRPC. The recommended simpler first implementation is a small `AsyncioTcpControlTransport` built on Python's `asyncio` streams. It needs no broker, RPC runtime, or generated client/server stubs and directly exposes the TCP-close signal Oobleck uses for failure detection. Keep membership and recovery logic behind a `ControlTransport` interface so transport mechanics cannot leak into the membership state machine. Replace Click with Tyro for all new CLI surfaces.

Use one persistent full-duplex TCP connection from each agent to the master. Encode each message as a four-byte big-endian length followed by a UTF-8 JSON envelope containing protocol version, message type, agent ID, incarnation ID, sequence number, membership generation, and a type-specific payload. Enforce a small maximum frame size and strict field/type validation. The master uses `asyncio.start_server()`; agents use `asyncio.open_connection()`; readers use `readexactly()` for the header and body; and each connection has one serialized writer queue that calls `drain()` for backpressure. EOF, reset, or an incomplete frame closes the incarnation and feeds the same lease/membership logic as heartbeat expiry.

The transport must define reconnect and duplicate-message semantics explicitly: a new incarnation replaces an old connection only through the membership state machine, sequence numbers reject duplicate/out-of-order messages, and generation numbers reject stale commands. Unit tests must fragment and coalesce frames, block writers to exercise backpressure, send oversized/malformed messages, reconnect an incarnation, and close several sockets concurrently. The old gRPC implementation may remain temporarily as a behavior reference during PR 5, but remove its runtime, generated files, and test dependencies after the asyncio transport reaches parity. TLS and authentication remain a separately scoped production-hardening concern.

Each agent:

- self-registers with a stable node identity, addresses, GPU inventory, and incarnation ID;
- maintains a bidirectional stream to the master;
- sends heartbeats (default every 1 second) under a renewable lease (default 5 seconds);
- treats TCP stream closure as an immediate failure signal;
- relays membership-generation notices to local GPU workers over a local IPC channel;
- supports an optional initial SSH hostfile, without requiring one for later joins.

The master maintains monotonically increasing membership generations. Abrupt failure, graceful drain, join, and replacement all produce a proposed generation. Workers reject stale messages and never combine rank maps from different generations.

Failure handling must operate on a set of failed node or agent identities, never assume exactly one failure. The master serializes concurrent disconnect/lease events against the live membership snapshot and coalesces events observed before a generation is published. If another agent fails while a generation is being prepared, torn down, or activated, the newer full-membership snapshot supersedes the in-progress generation. Workers abort the stale recovery attempt, tear down any partially initialized WORLD, and plan again from all currently surviving nodes. This covers simultaneous failures as well as cascading failures during recovery.

Application-level heartbeats, leases, generation numbers, and acknowledgements define correctness. gRPC/TCP keepalive and stream-close callbacks are detection inputs only; Oobleck must not depend on a transport-specific callback for membership consistency.

The master is a single point of failure in the first release. High availability, replicated control state, and control-plane TLS are follow-up work.

## 11. Step transaction and replay semantics

Treat each logical global batch as a transaction:

1. Assign deterministic sample indices and microbatch IDs.
2. Execute forward/backward under the active generation.
3. Complete required gradient synchronization.
4. Check that no newer membership generation was announced.
5. Atomically commit optimizer step, scheduler step, scaler update, and `committed_step`.

If a membership event is received before commit, ranks finish or unwind to the nearest safe schedule boundary, but must not update model parameters. Clear partial gradients, retire the generation, reconfigure from the newest complete membership snapshot, and replay the same `OobleckBatch`. Do not advance the optimizer, scheduler, scaler growth tracker, sampler cursor, epoch cursor, or global step. Multiple membership changes during the same attempt still cause only one replay after a viable generation becomes active.

Deterministic sample order must be represented by explicit indices, not by relying on a live iterator that cannot rewind. Derive stochastic model behavior from logical keys such as:

```text
(seed, epoch, committed_step, global_microbatch_id, global_layer_id)
```

This makes replay independent of the new physical pipeline and rank assignment.

### Dataset and DataLoader contract

The first release accepts any stable map-style `torch.utils.data.Dataset` implementing `__len__` and indexed `__getitem__`. A Hugging Face `datasets.Dataset` returned by `load_dataset(..., split="train")` is a primary supported case and must be passed directly as the dataset argument of a standard `torch.utils.data.DataLoader`. If `load_dataset()` returns a `DatasetDict`, require the caller to select a split such as `dataset["train"]`; reject it with an actionable message rather than treating split names as sample indices. Hugging Face streaming datasets and arbitrary `torch.utils.data.IterableDataset` instances require a durable cursor/checkpoint protocol and are deferred.

`OobleckParallelContext.create_batch_sampler()` returns an `OobleckBatchSampler` for use through the DataLoader's `batch_sampler=` argument. The sampler owns the deterministic epoch permutation, logical global-batch ID, and heterogeneous per-pipeline index assignment. Because PyTorch makes `batch_sampler` mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`, examples and validation must configure those policies through Oobleck instead of also passing them to `DataLoader`.

`context.prepare_dataloader(dataloader)` validates that the object is a standard `torch.utils.data.DataLoader` over a supported dataset and Oobleck sampler, then returns a thin ordered iterator adapter. The adapter pairs each DataLoader result with its sampler descriptor to produce `OobleckBatch`; it does not copy, convert, or replace the user's underlying dataset. Only a successful `context.step()` commits the sampler cursor. On failure, discard the DataLoader iterator and any prefetched but uncommitted descriptors, rebuild the iterator from the last committed cursor, and issue exactly the same sample indices under the new pipeline allocation.

Support `num_workers=0` and positive worker counts. Multi-worker loading must preserve result order, seed the DataLoader generator and workers from Oobleck's logical seed, and use `persistent_workers=False` in the first release so a reconfiguration can discard and deterministically recreate workers. The exact replay guarantee assumes stable indexed samples and a deterministic collator/transform. Random per-sample augmentation must use an Oobleck-provided logical sample seed or be materialized ahead of time; merely resetting a worker seed is not sufficient when the physical worker assignment changes.

The collator may tokenize raw Hugging Face rows and should follow Cornstarch's contract by returning either one batch dictionary or a list of microbatch dictionaries. Oobleck normalizes the result to a list of microbatches without requiring `.with_format("torch")`; users may still enable that format for already-tokenized numeric columns.

## 12. Exact process-group lifecycle

Preserve the current Oobleck behavior: every membership change destroys the entire distributed process-group universe, including WORLD, before a replacement world is initialized.

For an abrupt failure:

1. The CPU control plane notifies all surviving agents and workers.
2. Workers mark the current attempt uncommittable.
3. At the safe boundary, snapshot all known PyTorch process groups and backends.
4. Invoke backend shutdown concurrently so one blocked NCCL group cannot serialize teardown of the others.
5. Destroy WORLD.
6. Clear the same private c10d registries used by the existing implementation, isolated behind one version-checked compatibility module.
7. Verify that no stale Cornstarch meshes, schedules, or Oobleck subgroup handles remain.
8. Compile the next local ownership before initializing the next WORLD.

Graceful drain and join use the same generation boundary and complete teardown; they are not in-place group edits. Add explicit PyTorch-version guards and fail fast if internal registry layout changes, because silently retaining an old communicator is unsafe.

## 13. Reconfiguration and pipeline merge

Reimplement Section 5.1 and Figure 8(c) as a deterministic, pure planner. The planner accepts an arbitrary set of failed agents and must handle damage to several pipelines in one invocation. Given the previous instances, complete surviving-node set, supported templates, and target fault threshold, execute these phases in order:

1. **Remove failed ranks.** Remove every failed agent/rank atomically from the input snapshot and preserve the surviving members and logical pipeline identity where possible.
2. **Simple re-instantiation.** If a damaged pipeline still has a supported resource count, instantiate the best matching template directly.
3. **Borrow ranks.** For every undersized pipeline, borrow ranks from the largest donor pipeline while the donor remains at or above the minimum supported template size. Re-instantiate both affected pipelines.
4. **Merge pipelines.** If borrowing cannot make a valid pipeline, merge undersized pipelines. When necessary, consume a valid pipeline as a donor until every resulting group reaches a supported minimum. Select templates for the merged resource sets and remove obsolete pipeline identities.
5. **Validate.** Ensure all ranks are assigned once, all templates are supported, global batch allocation is feasible, and the requested replica/failure threshold is satisfied. Otherwise return `RecoveryUnavailable` rather than a partially valid plan.

Tie-breaking must be stable. Primary objective: minimize predicted iteration time. Secondary objective: maximize bytes of state retained in place. Further tie-breakers: minimize moved nodes, then choose lexicographically by stable node identity.

For joins and replacements, re-enumerate candidate template compositions up to `max_nodes`. Move to a faster valid composition only at a generation boundary. The same retained-state objective prevents gratuitous reshuffling when throughput estimates are equal.

## 14. All-to-all state redistribution

Replace the legacy tensor-by-tensor `send`/`recv` implementation with a deterministic all-to-all redistribution protocol.

### State included

- model parameters;
- persistent buffers;
- optimizer slots associated with each parameter;
- optimizer parameter-group metadata needed for faithful reconstruction;
- AMP/gradient-scaler state;
- scheduler state and committed-step metadata.

Gradients are not transferred because an interrupted attempt is discarded and replayed.

### Candidate discovery and transfer units

Compare the old and new manifests by stable logical key and TP lane. A source is eligible only if it survived, owns the exact shard/placement, and holds the same committed-step version as the destination plan. Parameters retained at the same rank are not communicated. For each missing parameter, keep the parameter, persistent buffer, and optimizer slots in one logical state bundle so source selection and validation cover the complete training state.

Split bundles into bounded transfer units. Small tensors are coalesced by destination and dtype; a tensor or bundle larger than the configured chunk size is divided into aligned chunks. Chunking is necessary to stripe a large layer across several equivalent replicas instead of making one source send the entire layer.

For standard PyTorch optimizers, serialize tensor slots independently inside the bundle and reconstruct scalar/non-tensor fields from versioned metadata. Initially reject custom optimizer state that cannot be described by the supported schema.

### Load-balanced source assignment

Do not select the first valid replica. Build the candidate source set for every transfer unit, sort units largest-first, and assign each unit to the candidate that minimizes the predicted recovery makespan. Track at least:

- bytes already assigned to every source GPU/node;
- bytes assigned to every destination GPU/node;
- bytes assigned to each known link class or failure-domain pair;
- source/destination bandwidth estimates and a locality cost.

For a homogeneous cluster without measured topology, the initial score is the maximum of projected source egress time and destination ingress time. This reduces to deterministic byte balancing and is substantially better than balancing tensor count. When topology data is available, add same-node/NVLink, same-rack, and cross-rack costs and score the projected bottleneck link as well. Prefer no-copy retention first, then locality, then the lowest projected makespan; use stable rank and logical-key ordering only as the final tie-breaker.

Allow different chunks of the same large tensor and different destination replicas of the same state to use different sources. Thus all surviving data-parallel replicas contribute concurrently, including when only one new destination needs a large state bundle. If there are more destinations than initial copies, optionally use verified destinations as relay sources in later collective rounds; enable this fan-out optimization only when the cost model predicts that the extra round is faster than repeated sends from the original candidates.

The planner emits an immutable transfer schedule containing source, destination, logical key, chunk range, dtype bucket, round, and expected byte count. Every rank derives or verifies the same schedule hash before communication. Record predicted and actual per-source bytes, per-destination bytes, link-class bytes, and round duration so later recoveries can update bandwidth estimates.

### Transport

After the new WORLD is live but before training resumes:

1. Exchange or verify the transfer-schedule hash, metadata, and split sizes.
2. Pack each round's assigned chunks into source/destination-specific dtype buckets.
3. Flatten aligned payloads into bounded-size buffers.
4. Use `torch.distributed.all_to_all_single` with variable input/output split sizes so all selected replica sources send concurrently.
5. Unpack directly into destination tensors where possible and make completed chunks eligible for a planned relay round.
6. Validate key, version, shape, dtype, shard placement, chunk range, byte count, and checksum/debug hash.

All ranks participate in the same ordered collective rounds with zero-sized splits when they have no payload. Bound each round's temporary memory and use double-buffered packing plus `async_op=True` only after correctness tests establish the required CUDA-stream synchronization. Isolate `all_to_all_single` behind a compatibility wrapper because PyTorch currently documents this API as experimental.

Point-to-point communication is not the default for the first implementation. P2P may reduce overhead for very sparse transfers, but it complicates ordering, liveness, and deadlock avoidance during recovery. The batched all-to-all protocol matches the explicit requirement and gives deterministic collective ordering. Add instrumentation so a later benchmark can justify a sparse P2P fast path if it materially improves recovery latency.

## 15. CLI and configuration

Introduce Tyro dataclasses for:

- master service;
- node agent;
- training launch;
- offline profiling/template generation;
- graceful drain and membership inspection;
- recovery/chaos test controls.

Keep configuration serializable and validate it before launching workers. Important validation includes fixed per-node TP width, model/template fingerprint compatibility, global-batch feasibility, fault-threshold feasibility, unique stable node IDs, supported PyTorch/Cornstarch/datasets versions, stable map-style dataset length/fingerprint, and rendezvous address reachability.

Legacy Click entry points may remain as deprecated wrappers for one transition release only if needed by existing tests; all new code and examples use Tyro.

## 16. End-to-end examples

Add an executable Oobleck example suite based on Cornstarch's `examples/distributed/pretrain_llm.py`, but split it by Oobleck role rather than pretending the whole system is one `torchrun` script:

- `examples/pretrain_llm.py`: builds the Cornstarch LLM, execution plan, optimizer, and scheduler; supports `FakeTextDataset` for fast offline runs and a map-style Hugging Face `datasets.Dataset` from `load_dataset(..., split="train")`; passes either dataset directly to `torch.utils.data.DataLoader` with Oobleck's batch sampler; uses `OobleckParallelizationPlan`/`OobleckParallelContext`; and leaves distributed initialization to Oobleck;
- `examples/run_master.py`: starts the membership/control master with Tyro configuration;
- `examples/run_agent.py`: self-registers one node, starts the local GPU workers, and runs the training entry point supplied by the master/job configuration;
- `examples/run_local.py`: starts a one-node development deployment with one master and one agent, suitable for the single-CUDA-GPU environment and smoke tests;
- `examples/drain_agent.py`: demonstrates a graceful generation change through the public control API;
- `examples/fail_agent.py`: opt-in failure-injection helper for a local/demo job, with explicit target validation so it cannot kill unrelated processes.

Keep the scripts thin: orchestration belongs in Oobleck library code, and the examples should demonstrate only public APIs. The pretraining example should retain Cornstarch's top-to-bottom structure—model construction, execution-plan construction, Hugging Face dataset loading, standard DataLoader construction, replayable sampler attachment, context materialization, optimizer/scheduler setup, and training step—while showing the agent-managed lifecycle and transactional `context.step()`. `FakeTextDataset` is an acceptable default for the main example so it remains fast, deterministic, and runnable offline. The same script must expose dataset name, configuration, and split arguments that switch to `load_dataset()` without changing the Oobleck or DataLoader path.

Add `examples/README.md` with:

- prerequisites, the `datasets` dependency, and the pinned Cornstarch dependency;
- synthetic-versus-Hugging-Face dataset selection, dataset name/configuration, explicit split selection, download/cache behavior, preprocessing/tokenization, and offline-cache usage;
- profile/template generation;
- exact start order and commands for master, initial agents, and training;
- a one-node single-GPU walkthrough;
- a multi-node launch walkthrough with addresses and stable node IDs;
- commands for simultaneous failure injection, graceful drain, and joining/replacement agents;
- expected logs for generation changes, WORLD teardown, state transfer, replay, and resume;
- cleanup instructions, troubleshooting, and a warning that the failure helper is for disposable example jobs.

Exercise the examples through local import/configuration tests and add a subprocess smoke test for the one-node flow. Multi-node/NCCL execution remains an explicitly documented manual or dedicated-runner test.

## 17. Pull-request sequence

### PR 1 — Branch bootstrap and dependency boundary

- Create the Oobleck `refactor` base branch.
- Add architecture decision records for scope, lifecycle, state identity, and replay semantics.
- Remove or quarantine obsolete ColossalAI integration from the new package path.
- Define dependency pinning to the Cornstarch PR head.
- Establish documented local formatting, type-checking, and unit-test commands for the refactor path; keep validation developer-run and repository-local.
- Copy Cornstarch's `tests/distributed/distributed_base.py:GlooDistributedTestBase` and `tests/distributed/gloo_utils.py` into the same paths in Oobleck, retaining source/license attribution and changing only package-local imports or necessary compatibility code.
- Add a multi-rank smoke test proving that model execution is on CUDA while distributed communication uses Gloo.

### Cornstarch PR — Compiled local partitions

- Create `refactor-oobleck` from Cornstarch `refactor`.
- Implement Sections 6.1 through 6.5.
- Add the Cornstarch acceptance tests from Section 6.6.
- Open a PR to Cornstarch `refactor`, leave it open, and pin Oobleck to its tested head SHA.

### PR 2 — Oobleck plan/context foundation

- Add configuration and public types.
- Add stable generation, rank, pipeline, stage, and logical-state identifiers.
- Wrap the new Cornstarch compile/activate lifecycle.
- Add `OobleckBatchSampler`, standard DataLoader attachment/validation, and the logical batch descriptor API.
- Add the Tyro launch/profile command skeletons.

### PR 3 — Profiling and template planning

- Port model/hardware profiling.
- Define versioned profile/template schemas and cache validation.
- Reuse the Rust template generator behind a narrow Python API; defer structural refactoring to `tasks/backlog.md`.
- Upgrade `pyo3` and `pyo3-build-config` to at least `0.24.1`, regenerate the lockfile, and update/test the binding API.
- Implement heterogeneous composition and global-batch allocation.

### PR 4 — Pure reconfiguration planner

- Implement set-based failure filtering, simple re-instantiation, borrowing, and merge for single and simultaneous failures.
- Add join, replacement, and graceful-drain planning.
- Add deterministic throughput/state-retention objectives and exhaustive planner tests.

### PR 5 — Elastic control plane

- Refactor the master/agent protocol around complete membership snapshots and set-based generations.
- Define `ControlTransport` and implement the recommended length-prefixed `AsyncioTcpControlTransport`.
- Add self-registration, heartbeats, leases, TCP-close detection, reconnect/sequence semantics, local worker notifications, and drain/join operations.
- Keep gRPC only as a temporary parity reference, then remove `grpcio`, generated stubs, and gRPC-specific test dependencies.
- Replace Click-based commands with Tyro.
- Unit-test framing, partial reads, backpressure, malformed frames, reconnects, concurrent disconnects, and the transport-independent membership state machine.

### PR 6 — Initial heterogeneous execution

- Compile rank-local ownership before WORLD.
- Build heterogeneous per-pipeline PP/TP meshes after WORLD initialization.
- Activate Cornstarch local partitions and 1F1B schedules.
- Build logical-layer/TP-lane gradient synchronization groups.
- Validate heterogeneous batch weighting and checkpoint initialization.

### PR 7 — Process-group lifecycle and transactional steps

- Isolate and port the exact force-shutdown implementation.
- Add generation abort, safe-boundary handling, gradient discard, and batch replay.
- Add deterministic map-style dataset sampling, DataLoader prefetch invalidation, committed-cursor replay, and logical RNG handling.

- Verify repeated teardown/reinitialization does not retain stale group state.

### PR 8 — All-to-all state transfer and full churn

- Implement manifest diffing, candidate discovery, chunking, and deterministic makespan-aware source assignment.
- Stripe large state bundles across surviving replicas and optionally use cost-model-approved relay rounds.
- Transfer parameters, buffers, optimizer, scaler, scheduler, and committed-step state through balanced all-to-all rounds.
- Integrate single/simultaneous failure, failure-during-recovery, drain, join, and replacement recovery end to end.
- Add metrics for source/destination/link utilization, scheduling error, straggler round, detection, planning, teardown, transfer, activation, and total recovery latency.

### PR 9 — End-to-end examples and operator walkthrough

- Implement the role-specific scripts and `examples/README.md` from Section 16. The LLM example supports both `FakeTextDataset` and `load_dataset(..., split="train")` through the same standard `torch.utils.data.DataLoader` construction.

- Add import/configuration tests and a single-node subprocess smoke test.
- Validate failure, drain, replay, and resume instructions against the example code.

### PR 10 — Research validation and operator documentation

- Add multi-node NCCL chaos tests and long-running churn tests.
- Reproduce simple, borrow, and merge recovery examples from the paper, including simultaneous failures affecting multiple pipelines.
- Add deployment, profiling, debugging, compatibility, and failure-mode documentation.
- Publish benchmark scripts and machine-readable results for steady-state and recovery overhead.

## 18. Test and validation strategy

### Unit tests

Every implementation PR must add focused unit tests in the same PR; tests are deliverables, not a final hardening phase. At minimum cover:

- template serialization and compatibility fingerprints;
- batch allocation and gradient weights;
- every simple/borrow/merge branch, including impossible recovery;
- simultaneous failures in one pipeline and across multiple pipelines;
- a newer failure superseding an in-progress recovery generation;
- deterministic planner tie-breaking;
- state-manifest matching, candidate filtering, chunking, balanced source assignment, schedule hashing, bucketing, and reconstruction;
- deterministic source balance by bytes, including a single large tensor striped across several replicas;
- topology/locality scoring, stale-version exclusion, and optional relay-round selection;
- membership-generation transitions and stale-message rejection;
- replayable sampler, committed cursor, prefetched-batch invalidation, and logical RNG derivation;
- plain PyTorch Dataset and Hugging Face `datasets.Dataset.from_dict` compatibility;
- actionable rejection of `DatasetDict`, streaming/iterable datasets, invalid DataLoader arguments, and nondeterministic worker configurations.

### Single-GPU distributed unit-test harness

Copy Cornstarch's `GlooDistributedTestBase` and `gloo_utils.py` into `tests/distributed/`. The copied base uses PyTorch's `MultiProcessTestCase` to spawn several ranks, initializes `torch.distributed` with Gloo, maps every rank to the available CUDA device using `rank % torch.cuda.device_count()`, disables Dynamo/cuDNN nondeterminism, and patches unsupported or CPU-only Gloo collective paths. Copy the monkeypatches for `batch_isend_irecv`, `all_to_all`, `all_to_all_single`, `reduce_scatter`, and `all_gather`, including subgroup/global-rank translation and variable split-size handling.

These are GPU execution tests, not CPU model tests: models, parameters, activations, losses, and optimizer state remain on CUDA. Only collective payloads are staged through CPU where Gloo requires it and copied back to CUDA by the monkeypatches. Tests must assert CUDA device placement and Gloo backend so an accidental CPU fallback fails visibly.

This development node has one CUDA GPU. Before running distributed tests, check both `nvidia-smi -L` and `torch.cuda.is_available()`/`torch.cuda.device_count()`. A transient failure to detect or initialize CUDA must be retried until the GPU becomes available; it must not cause the suite to skip, xfail, or silently run the distributed feature on CPU.

Use this GPU-over-Gloo base for distributed unit tests in every relevant PR, including:

- heterogeneous pipeline layouts with small synthetic models;
- all-to-all variable split sizes, including zero-payload ranks and several concurrent sources;
- balanced redistribution that reconstructs byte-identical model/optimizer state while bounding the maximum source load;
- model and optimizer state movement after merge/split;
- standard DataLoader execution over a Hugging Face `datasets.Dataset`, with heterogeneous sample assignment and no duplicate committed indices;
- abort before commit and exact replay of sample IDs with `num_workers=0` and a positive worker count;
- repeated WORLD teardown/recreation;
- simultaneous rank-loss plans and a failure that supersedes recovery;
- CUDA numerical equivalence against a local reference.

After teardown, assert that PyTorch's relevant process-group registries are empty and all Oobleck/Cornstarch context handles are retired.

### CPU control-plane tests

Master/agent protocol tests that do not execute training may run on CPU. Cover concurrent stream closure, lease expiry, event coalescing, stale generation rejection, join, replacement, and graceful drain without requiring CUDA collectives.

### GPU tests

- the mandatory single-CUDA-GPU, multi-rank Gloo suite described above;
- single-node, multi-GPU NCCL smoke tests when a suitable runner is available;
- DTensor TP correctness and state-manifest stability;
- heterogeneous PP/TP gradient equivalence against a non-pipelined reference;
- optimizer/scaler/scheduler equivalence across a recovery;
- recovery-latency and maximum-source-load comparison against the legacy first-source policy;
- real multi-node NCCL abrupt-failure tests where the control plane triggers recovery without waiting for NCCL timeout;
- simultaneous multi-agent failure and cascading failure-during-recovery tests;
- pipeline merge followed by successful replay and continued loss convergence;
- repeated churn under memory-leak and communicator-leak monitoring.

### Research acceptance criteria

- Steady-state updates match a reference run within the chosen numerical tolerance.
- A failure before commit never advances logical training state.
- The interrupted global batch is replayed exactly once after recovery.
- A Hugging Face `datasets.Dataset` selected by `load_dataset(..., split=...)` runs through a standard `torch.utils.data.DataLoader`, and recovery neither skips nor duplicates committed sample indices.
- Simple, borrow, and merge paths are all observed in end-to-end tests.
- Several agents can fail in the same detection window or during recovery without committing a partial step or activating a stale generation.
- Every membership change creates a new WORLD and leaves no usable old communicator.
- All required state reaches its new owner with no checkpoint reload when a surviving replica has it.
- Failure detection latency is governed by TCP close/lease settings, not NCCL timeout.
- Recovery and throughput metrics are sufficient to reproduce the paper-style evaluation.

## 19. Compatibility and staged rollout

Gate the new implementation behind the refactor package/API until it passes parity tests. Do not mix legacy and new engine objects in one job. Record and validate a compatibility tuple containing Oobleck commit, Cornstarch commit, PyTorch version, CUDA/NCCL version, model fingerprint, template schema version, optimizer schema version, `datasets` version, dataset/preprocessing fingerprint, and hardware fingerprint.

Roll out in this order:

1. single pipeline, no failure, DP/PP/TP correctness;
2. multiple homogeneous replicas;
3. heterogeneous templates and cross-replica synchronization;
4. graceful generation replacement;
5. abrupt failure with simple recovery;
6. borrow recovery;
7. merge recovery;
8. simultaneous and cascading failures;
9. joins/replacements and repeated churn.

## 20. Explicit assumptions and non-goals

- One agent/node is one failure domain and contributes one fixed-size TP group to a pipeline stage.
- Nodes are homogeneous for the first release; heterogeneous stage speed comes from layer assignment, not mixed GPU types.
- Fault tolerance threshold `f` requires at least `f + 1` viable pipeline replicas when resources permit.
- Only Cornstarch LLM families with stable repeated-layer identities are initially supported.
- Map-style datasets and standard PyTorch optimizer state are supported first.
- WORLD is always fully replaced on membership change; in-place elastic process-group edits are not considered.
- The master service is not highly available in the first release.
- CP and EP support are deferred to `tasks/backlog.md`; multimodal models, elastic TP width, arbitrary streaming-dataset replay, and operation beyond configured `max_nodes` are also out of scope for the first implementation.

## 21. Definition of done

The refactor is complete when an Oobleck job based on the pinned Cornstarch `refactor-oobleck` commit can start from meta model ownership without WORLD, run heterogeneous pipeline replicas, detect one or multiple simultaneous node losses through the CPU control plane, abort the uncommitted step, fully destroy WORLD, choose a configuration using simple/borrow/merge, recreate WORLD, redistribute all missing training state using all-to-all, replay the same logical global batch, and continue training with numerically correct updates. A newer failure during recovery must safely supersede the in-progress generation. The same generation mechanism must also handle graceful drains and joining or replacement nodes, with distributed unit tests on the single CUDA GPU over Gloo and documented end-to-end examples for each path.
