# Oobleck

Oobleck is a heterogeneous elastic pipeline-training runtime built around
Cornstarch's plan/context model. It chooses rank-local ownership before WORLD,
uses checksummed membership generations, fully replaces the process-group
universe on every change, and treats each logical global batch as a replayable
transaction.

```python
plan = OobleckParallelizationPlan(
    OobleckConfig(global_batch_size=128, microbatch_size=2, max_nodes=16)
)
plan.parallelize(model, ParallelConfig(tensor_parallel_size=8))
context = plan.materialize("cuda", dtype=torch.bfloat16)

sampler = context.create_batch_sampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, persistent_workers=False)
loader = context.prepare_dataloader(dataloader)
context.configure_optimization(
    optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=1e-4)
)
for batch in loader:
    context.step(batch, execution_plan, output, criterion)
```

The first release supports stable map-style datasets and fixed per-node tensor
parallel width. Oobleck selects heterogeneous pipeline and effective data
parallel layouts. Context/expert parallelism, streaming-dataset replay, elastic
TP width, multimodal models, and master high availability are out of scope.

See [the examples](examples/README.md), the architecture decisions in
[`docs/adr/`](docs/adr/), and [`tasks/plan.md`](tasks/plan.md) for the full
design and acceptance contract.

## Runtime lifecycle

Oobleck separates the CPU control plane from the distributed training data
plane. Its lifecycle has six stages: offline profiling and template generation,
node registration and ownership compilation, distributed initialization,
transactional training, failure/event propagation, and worker reconfiguration
with committed-state transfer.

See the [runtime lifecycle guide](lifecycle.md) for the complete walkthrough,
architecture diagrams showing where the master, node agents, and GPU workers
run, their communication paths, the template-cache format, and the relationship
between heterogeneous composition and the paper algorithm.

## Package directory guide

The import without a suffix is the authoritative public/current implementation.
Several areas deliberately keep a lower-level `*_base.py` module containing
core value types or mechanics. Some public modules import and extend that base
directly; in other pairs the base is a retained lower-level reference and the
public module contains the current implementation. When tracing a code path,
start at the public module and follow its imports. `state.py` has one additional
active intermediate layer, `state_public_base.py`.

### Top-level `oobleck/` modules

| File | Responsibility |
| --- | --- |
| [`__init__.py`](oobleck/__init__.py) | Defines the supported top-level public API: plans, contexts, batches, state manifests, configuration, and compatibility helpers. |
| [`acceptance.py`](oobleck/acceptance.py) | Emits JSONL runtime/recovery metrics and verifies replay, strategy coverage, CUDA-memory growth, process-group growth, and clean shutdown for optional chaos/churn runs. |
| [`cli.py`](oobleck/cli.py) | Tyro command dispatcher for the master, agent, training launch, profiling, membership inspection, drain, and guarded chaos helper. |
| [`compatibility.py`](oobleck/compatibility.py) | Builds model, dataset/preprocessing, package revision, CUDA/NCCL, and hardware fingerprints used for cross-worker generation consensus. |
| [`config.py`](oobleck/config.py) | Serializable command configurations for master/agent services, launch, profiling, drain, inspection, and chaos injection. The training runtime's `OobleckConfig` lives in `types.py`. |
| [`types.py`](oobleck/types.py) | Immutable public value types: templates, stage specs, pipeline instances, execution plans, runtime compatibility, checksums, rank maps, and runtime configuration. |
| [`runtime_base.py`](oobleck/runtime_base.py) | Core `OobleckParallelizationPlan`, prepared/context objects, initial plan compilation/materialization, DataLoader attachment, and logical step transaction. |
| [`runtime.py`](oobleck/runtime.py) | Public runtime layer that extends the base context with membership application, two-phase generation replacement, supersession, state recovery, generation barriers, transition metrics, and complete topology-aware close. |
| [`cornstarch_base.py`](oobleck/cornstarch_base.py) | Retained minimal compile/activate adapter and fallback local-manifest reference for the pinned Cornstarch lifecycle. The active runtime imports `cornstarch.py`. |
| [`cornstarch.py`](oobleck/cornstarch.py) | Current Cornstarch boundary used by the runtime; preserves the execution plan through compilation, imports Cornstarch manifests, creates heterogeneous meshes on activation, and retires external contexts idempotently. |
| [`data_base.py`](oobleck/data_base.py) | Deterministic logical batch descriptors, committed-cursor sampler, microbatch slicing, prefetch invalidation, replay, and the prepared DataLoader adapter. |
| [`data.py`](oobleck/data.py) | Public DataLoader facade; re-exports the base types and adds strict validation that sampler and DataLoader use the exact same dataset object. |
| [`meshes.py`](oobleck/meshes.py) | Creates deterministic per-pipeline Cornstarch `DeviceMesh` objects from an execution plan's heterogeneous stage/rank layout. |
| [`topology.py`](oobleck/topology.py) | Derives cross-pipeline gradient groups from logical parameter identity and TP lane, creates the process groups, applies sample-weighted synchronization, and closes the groups. |
| [`optimization_base.py`](oobleck/optimization_base.py) | Retained lower-level versioned optimizer-schema and logical-key serialization/restore reference. The active recovery path imports `optimization.py`. |
| [`optimization.py`](oobleck/optimization.py) | Current optimizer-state implementation, including partition-manifest keys, local DTensor shard extraction, optimizer-group reconstruction, and device/dtype-aware slot restore. |
| [`state_base.py`](oobleck/state_base.py) | Defines logical state entries, manifests, transfers, immutable schedule checksums, candidate/version validation, chunking, round packing, locality classification, and balanced source selection. |
| [`state_public_base.py`](oobleck/state_public_base.py) | Intermediate public planner that gives each homogeneous source/destination link an independent load class, then normalizes reported link metrics. |
| [`state.py`](oobleck/state.py) | Final public state API; supplies topology-free bandwidth defaults so destination ingress does not hide the source-load balancing tie-break. This is the module runtime/recovery code imports. |
| [`state_transfer.py`](oobleck/state_transfer.py) | Packs tensors into the planned byte ranges, executes collective all-to-all rounds, validates dtype/checksum/reconstruction, and returns transfer timing/load metrics. |
| [`recovery_base.py`](oobleck/recovery_base.py) | Captures committed model/buffer/optimizer/scheduler/scaler state, builds combined manifests and logical tensor bindings, copies retained state, and defines recovery snapshots/reports. |
| [`recovery.py`](oobleck/recovery.py) | Public recovery coordinator; gathers old and new manifests across ranks, plans every destination together, executes transfer schedules, restores optimizer/training metadata, and reports planning/transfer balance. |

### `oobleck/planning/`

| File | Responsibility |
| --- | --- |
| [`__init__.py`](oobleck/planning/__init__.py) | Re-exports the supported profiling, template, composition, and reconfiguration API. |
| [`profiler.py`](oobleck/planning/profiler.py) | Measures model layers with a model-agnostic workload and records versioned execution/memory profiles. |
| [`generator.py`](oobleck/planning/generator.py) | Narrow binding around the exact Rust minimax template generator with an equivalent deterministic Python fallback; see [`docs/pipeline_planner.md`](docs/pipeline_planner.md). |
| [`planner.pyi`](oobleck/planning/planner.pyi) | Type stub for the compiled Rust `create_pipeline_templates` extension. |
| [`cache.py`](oobleck/planning/cache.py) | Saves and loads versioned templates while validating schema and compatibility fingerprints. |
| [`composer.py`](oobleck/planning/composer.py) | Selects a deterministic heterogeneous template composition, maps it to nodes, allocates global microbatches, and computes gradient sample weights. |
| [`reconfiguration.py`](oobleck/planning/reconfiguration.py) | Pure membership replanner implementing simple re-instantiation, node borrowing, pipeline merge, joins/replacements, and deterministic state-retention/throughput objectives. |

### `oobleck/distributed/`

| File | Responsibility |
| --- | --- |
| [`__init__.py`](oobleck/distributed/__init__.py) | Exports replacement-WORLD initialization, full teardown, and the guarded layout error. |
| [`lifecycle.py`](oobleck/distributed/lifecycle.py) | Active process-group lifecycle: validates deterministic rendezvous inputs, initializes replacement WORLD, concurrently shuts down every known group, destroys WORLD, clears version-audited c10d registries, and verifies retirement. |
| [`lifecycle_base.py`](oobleck/distributed/lifecycle_base.py) | Retained lower-level teardown reference for registry clearing. The active `lifecycle.py` adds replacement initialization and explicit PyTorch-version/layout guards. |

### `oobleck/elastic/`

| File | Responsibility |
| --- | --- |
| [`__init__.py`](oobleck/elastic/__init__.py) | Re-exports the public membership, transport, local-worker, service, and hostfile API. |
| [`membership.py`](oobleck/elastic/membership.py) | Transport-independent membership state machine: stable identities/incarnations, registration, heartbeats, acknowledgements, leases, disconnects, drains, event coalescing, stale-message rejection, and snapshot hashing. |
| [`transport.py`](oobleck/elastic/transport.py) | Strict JSON `MessageEnvelope` schema and bounded length-prefixed asyncio transport, including partial-read handling, serialized writes, and backpressure. |
| [`service_base.py`](oobleck/elastic/service_base.py) | Master generation protocol and base agent stream: registration, heartbeat/lease processing, membership proposals, prepared-plan consensus, rendezvous publication, ready consensus, activation, drain, and inspection. |
| [`service_public_base.py`](oobleck/elastic/service_public_base.py) | Current reconnecting `NodeAgentClient`, local-worker phase aggregation, master sequence validation, TCP-close handling, and public inspect/status/drain helpers. |
| [`service.py`](oobleck/elastic/service.py) | Public service facade; combines the current client/helpers with the master and bounds broadcasts so a failed writer becomes a membership event instead of stalling recovery. |
| [`local_ipc.py`](oobleck/elastic/local_ipc.py) | Unix-domain relay between one CPU agent and its GPU workers. Aggregates every local worker's prepared/ready metadata and enforces membership, rendezvous, and active phase ordering. |
| [`workers.py`](oobleck/elastic/workers.py) | Agent-owned process supervisor that launches one training worker per configured GPU, assigns stable environment identity/rank data, and retires workers on completion or failure. |
| [`hostfile.py`](oobleck/elastic/hostfile.py) | Parses an optional bootstrap-only host inventory and constructs explicit SSH agent commands; later joins do not depend on the hostfile. |

## Suggested reading paths

- **Initial configuration:** `planning/profiler.py` → `planning/generator.py` →
  `planning/composer.py` → `runtime_base.py` → `cornstarch.py` →
  `distributed/lifecycle.py`.
- **Training and commit semantics:** `data_base.py` → `runtime_base.py` →
  `topology.py`.
- **Failure/event propagation:** `elastic/membership.py` →
  `elastic/service.py` → `elastic/local_ipc.py` → `runtime.py`.
- **Worker reconfiguration and data copy:** `runtime.py` →
  `distributed/lifecycle.py` → `planning/reconfiguration.py` → `recovery.py`
  → `state.py` → `state_transfer.py`.

## Development

```bash
pytest -q
cargo test
python examples/run_local.py
python benchmarks/recovery_schedule.py
python benchmarks/pipeline_planner.py
python benchmarks/transaction_overhead.py
```

The Rust planner binding requires PyO3 and `pyo3-build-config` 0.24.1 or newer;
its exact transitive versions are recorded in `Cargo.lock`.
