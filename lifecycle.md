# Oobleck runtime lifecycle

[Back to the main README](README.md).

Oobleck separates the CPU control plane from the distributed training data
plane. The control plane decides which complete membership generation may run;
the data plane compiles ownership, creates that generation's process groups,
executes transactional steps, and moves committed state when membership
changes.

```mermaid
flowchart TD
    P["Profile layers"] --> T["Generate and cache pipeline templates"]
    T --> M["Receive complete membership snapshot"]
    M --> C["Compose execution plan and rank map"]
    C --> L["Compile rank-local ownership without WORLD"]
    L --> PB["Prepared barrier: agree on snapshot, plan, compatibility"]
    PB --> W["Create replacement WORLD and activate local partition"]
    W --> R["Restore committed state when recovering"]
    R --> AB["Ready/active barriers"]
    AB --> S["Run transactional training steps"]
    S --> F{"Membership changed?"}
    F -- "No" --> S
    F -- "Failure, drain, join, replacement" --> NM["Receive newer complete membership snapshot"]
    NM --> RP["Reconfigure with simple / borrow / merge"]
    RP --> X["Capture committed state and invalidate prefetch"]
    X --> D["Close schedules/meshes and destroy all process groups"]
    D --> L
```

### 1. Profile the model and generate pipeline templates

```mermaid
flowchart LR
    subgraph offline["Offline profiling process — no WORLD"]
        Model["Meta/model blueprint"] --> Profiler["ModelProfiler"]
        Workload["Representative inputs and loss"] --> Profiler
        Profiler --> Profile["Versioned ModelProfile JSON"]
        Profile --> Generator["Rust template generator<br/>or deterministic Python fallback"]
        Generator --> Templates["PipelineTemplate set<br/>for supported node counts"]
        Templates --> Cache["One versioned template-cache JSON file"]
    end
    Cache --> WA["Node A GPU worker<br/>load and validate fingerprint"]
    Cache --> WB["Node B GPU worker<br/>load and validate fingerprint"]
    WA --> CA["Deterministic composer"]
    WB --> CB["Deterministic composer"]
```

[`planning/profiler.py`](oobleck/planning/profiler.py) measures each repeated
layer's execution time, activation memory, and persistent state. The resulting
versioned `ModelProfile` can be saved and loaded independently of a running
distributed job. [`planning/generator.py`](oobleck/planning/generator.py) turns
those layer measurements into candidate `PipelineTemplate` objects through the
Rust planner extension, with a deterministic Python fallback.

#### Template cache format

[`planning/cache.py`](oobleck/planning/cache.py) **writes** one pretty-printed,
versioned JSON document to the caller-supplied path. The file contains one
compatibility fingerprint and a `templates` array, so a compatible set of
one-node, two-node, and larger templates normally shares one cache file; it is
not one template per file. `load_templates()` rejects a schema or fingerprint
mismatch before returning any template.

A shortened two-template cache looks like this:

```json
{
  "fingerprint": {
    "cornstarch_version": "71cf4ad681cfefdbf4a5b61d524af9033bebd055",
    "dtype": "bfloat16",
    "hardware": "NVIDIA-H100-80GB",
    "model": "gpt2-xl",
    "schema_version": 1,
    "tensor_parallel_size": 8
  },
  "schema_version": 1,
  "templates": [
    {
      "activation_memory": 2147483648,
      "backward_time": 0.24,
      "communication_time": 0.01,
      "fingerprint": {
        "cornstarch_version": "71cf4ad681cfefdbf4a5b61d524af9033bebd055",
        "dtype": "bfloat16",
        "hardware": "NVIDIA-H100-80GB",
        "model": "gpt2-xl",
        "schema_version": 1,
        "tensor_parallel_size": 8
      },
      "forward_time": 0.12,
      "layer_ranges": [[0, 48]],
      "max_microbatches": 16,
      "persistent_memory": 12884901888,
      "schema_version": 1,
      "template_id": "gpt2-xl-stages-1",
      "tensor_parallel_size": 8
    },
    {
      "activation_memory": 1288490188,
      "backward_time": 0.14,
      "communication_time": 0.02,
      "fingerprint": {
        "cornstarch_version": "71cf4ad681cfefdbf4a5b61d524af9033bebd055",
        "dtype": "bfloat16",
        "hardware": "NVIDIA-H100-80GB",
        "model": "gpt2-xl",
        "schema_version": 1,
        "tensor_parallel_size": 8
      },
      "forward_time": 0.07,
      "layer_ranges": [[0, 24], [24, 48]],
      "max_microbatches": 24,
      "persistent_memory": 7516192768,
      "schema_version": 1,
      "template_id": "gpt2-xl-stages-2",
      "tensor_parallel_size": 8
    }
  ]
}
```

The numeric values above are illustrative; the profiler and generator supply
them for the selected model and hardware.

#### How heterogeneous composition is chosen

[`planning/composer.py`](oobleck/planning/composer.py) does not choose one
single template. Given the current node count, global microbatch count, and
fault-tolerance threshold, it:

1. enumerates every multiset of templates whose `resource_count` exactly covers
   the available nodes and contains at least `fault_tolerance_threshold + 1`
   pipeline replicas;
2. assigns concrete sorted node IDs to each candidate; at a generation boundary
   it first maximizes bytes of state retained in place and then minimizes moved
   surviving nodes;
3. gives every pipeline at least one microbatch, respects each template's
   `max_microbatches`, and finds the smallest predicted makespan that can hold
   the complete global batch;
4. selects the candidate with the lexicographic objective
   `(maximum predicted iteration time, -retained state bytes, moved nodes,
   template IDs, stable pipeline/node IDs)`.

This is based on the [Oobleck paper's Section 4.2 pipeline-instantiation
algorithm](https://insujang.github.io/assets/pdf/sosp23_oobleck.pdf): enumerate
feasible template combinations, distribute the global batch, and select the
fastest plan. It is a deterministic reimplementation rather than a line-for-line
port of the SOSP artifact. The current code minimizes predicted makespan directly
and uses `PipelineTemplate.iteration_time()` (including its serialized
communication time), instead of the artifact's Pyomo variance objective and
separate all-reduce lookup. It also adds memory-capacity checks and the
retained-state/movement/stable-ID tie-breaks needed for elastic generations.
The paper's Section 5.1 simple/borrow/merge failure algorithm is implemented
separately in
[`planning/reconfiguration.py`](oobleck/planning/reconfiguration.py).

### 2. Register nodes and compile ownership before WORLD

The master and node agents communicate only through the CPU control plane in
[`elastic/`](oobleck/elastic/). An agent registers its stable node ID,
incarnation, reachable addresses, and GPU IDs, then maintains a heartbeat lease.
The master publishes every join, replacement, drain, disconnect, or lease
expiry as a complete, monotonically newer `MembershipSnapshot`.

```mermaid
sequenceDiagram
    participant M as CPU master process
    participant AA as Node A CPU agent
    participant AW as Node A GPU workers
    participant BA as Node B CPU agent
    participant BW as Node B GPU workers

    AA->>M: register(node ID, incarnation, addresses, GPU IDs)
    BA->>M: register(node ID, incarnation, addresses, GPU IDs)
    AA->>M: periodic heartbeat + lease generation
    BA->>M: periodic heartbeat + lease generation
    M-->>AA: complete membership snapshot + hash
    M-->>BA: complete membership snapshot + hash
    AA-->>AW: membership over Unix-domain socket
    BA-->>BW: membership over Unix-domain socket
    par Every GPU worker, without WORLD
        AW->>AW: compose plan and compile local ownership
        BW->>BW: compose plan and compile local ownership
    end
    AW-->>AA: worker_ack(prepared, plan checksum, compatibility)
    BW-->>BA: worker_ack(prepared, plan checksum, compatibility)
    AA-->>M: generation_prepared
    BA-->>M: generation_prepared
    Note over M: Publish rendezvous only after all agents agree
```

Each GPU worker receives the snapshot from its local agent over the Unix-domain
relay in [`elastic/local_ipc.py`](oobleck/elastic/local_ipc.py). It gives the
membership to `OobleckParallelizationPlan`, which:

1. uses [`planning/composer.py`](oobleck/planning/composer.py) for the initial
   layout or [`planning/reconfiguration.py`](oobleck/planning/reconfiguration.py)
   for a simple, borrow, or merge recovery;
2. creates a stable node/rank map and an immutable, checksummed
   `OobleckExecutionPlan`;
3. calls [`cornstarch.py`](oobleck/cornstarch.py) to compile the rank-local
   stage and logical state manifest without initializing `torch.distributed` or
   allocating real parameter storage.

Workers report the snapshot hash, plan checksum, and runtime compatibility
digest through the **prepared** barrier. The master publishes rendezvous data
only after every agent agrees on all three values.

### 3. Initialize distributed execution

After the prepared barrier,
[`distributed/lifecycle.py`](oobleck/distributed/lifecycle.py) creates exactly
one replacement WORLD from the generation's deterministic rank map and
rendezvous address. [`cornstarch.py`](oobleck/cornstarch.py) then activates the
compiled local partition, materializes local state, and constructs the
per-pipeline meshes from [`meshes.py`](oobleck/meshes.py).

```mermaid
sequenceDiagram
    participant M as CPU master process
    participant AA as Node A CPU agent
    participant AW as Node A GPU workers
    participant BA as Node B CPU agent
    participant BW as Node B GPU workers

    M-->>AA: generation_rendezvous(snapshot, plan, compatibility)
    M-->>BA: generation_rendezvous(snapshot, plan, compatibility)
    AA-->>AW: rendezvous over local IPC
    BA-->>BW: rendezvous over local IPC
    par Replacement WORLD creation
        AW->>BW: torch.distributed rendezvous and process-group creation
        BW->>AW: symmetric WORLD/mesh participation
    end
    AW->>AW: activate Cornstarch partition and local meshes
    BW->>BW: activate Cornstarch partition and local meshes
    AW-->>AA: worker_ack(ready)
    BW-->>BA: worker_ack(ready)
    AA-->>M: generation_ready
    BA-->>M: generation_ready
    M-->>AA: generation_active
    M-->>BA: generation_active
    AA-->>AW: release training-step barrier
    BA-->>BW: release training-step barrier
```

[`topology.py`](oobleck/topology.py) builds additional gradient synchronization
groups by stable logical layer identity and TP lane, allowing parameters shared
across heterogeneous pipeline replicas to receive correctly weighted global
gradients. Each worker reports **ready** only after activation and any recovery
finish. The master sends **generation active** only when every agent is ready;
managed training steps remain blocked until that final barrier.

### 4. Execute a transactional training step

[`data_base.py`](oobleck/data_base.py) assigns explicit sample indices and
global microbatch IDs to every logical batch. DataLoader prefetch may run ahead,
but `OobleckBatchSampler.committed_cursor` advances only when the step commits.
Logical RNG seeds depend on the batch and microbatch identity rather than the
physical rank, so replay uses the same data and randomness after reconfiguration.

```mermaid
flowchart LR
    DL["Prepared DataLoader"] --> B["OobleckBatch<br/>explicit sample and microbatch IDs"]
    B --> LP["Select this pipeline's microbatches"]
    LP --> CS["Cornstarch pipeline schedule<br/>forward + backward"]
    CS --> GS["Logical-layer / TP-lane<br/>gradient synchronization"]
    GS --> G{"New membership pending<br/>before commit?"}
    G -- "No" --> COMMIT["Atomically step optimizer/scaler,<br/>scheduler, committed step, sampler cursor"]
    G -- "Yes" --> DISCARD["Zero/discard gradients<br/>keep committed cursor unchanged"]
    DISCARD --> REPLAY["Replay the same logical batch<br/>after generation_active"]
```

`OobleckParallelContext.step()` in
[`runtime_base.py`](oobleck/runtime_base.py) executes only the microbatches
assigned to the local pipeline, synchronizes Cornstarch and heterogeneous
replica gradients, and then atomically advances:

- optimizer and optional gradient scaler;
- scheduler;
- committed training step;
- DataLoader sampler cursor.

If a membership change becomes pending before that atomic commit, Oobleck
discards the attempt's gradients and does not advance any of those objects.

### 5. Handle membership changes

Failures, joins, graceful drains, and replacements all enter the same generation
protocol. The master converts each event into a complete, monotonically newer
`MembershipSnapshot`; it never sends rank-local membership patches. Workers
first agree on that snapshot and a deterministic execution plan, then replace
the distributed generation through the common path in Section 5.3.

If another event arrives while a generation is being prepared or activated, the
newer complete snapshot supersedes the stale generation. Workers abandon the
stale plan, close any partially initialized distributed state, and prepare only
the newest membership.

#### 5.1. Detect failures and publish reduced membership

[`elastic/service.py`](oobleck/elastic/service.py) and
[`elastic/service_public_base.py`](oobleck/elastic/service_public_base.py) run
the reconnecting agent streams. A TCP close is an immediate failure signal;
otherwise [`elastic/membership.py`](oobleck/elastic/membership.py) expires the
agent heartbeat lease. The master removes the failed incarnation and publishes
the complete reduced live set.

```mermaid
sequenceDiagram
    participant FA as Failed node agent
    participant M as CPU master process
    participant SA as Surviving node agents
    participant SW as Surviving GPU workers

    alt TCP stream closes
        FA-xM: connection reset / EOF
        M->>M: remove failed incarnation immediately
    else Heartbeats stop
        M->>M: lease expires and removes agent
    end
    M->>M: coalesce live set and publish generation N+1
    M-->>SA: complete reduced membership snapshot N+1
    SA-->>SW: snapshot N+1 over local IPC
    SW->>SW: apply_membership and announce pending recovery plan
    Note over SW: An in-flight step cannot commit after the pending generation is visible
    opt Another failure, join, drain, or replacement arrives
        M->>M: publish generation N+2
        M-->>SA: newer complete snapshot N+2
        SA-->>SW: N+2 supersedes prepared or activating N+1
    end
```

The failure event path is:

```text
master membership state machine
  -> complete reduced membership snapshot over TCP
  -> surviving node agent
  -> local Unix-domain relay
  -> every local GPU worker
  -> OobleckParallelContext.apply_membership()
```

A failed worker may be required by the current pipeline schedule or collective,
so the in-flight logical batch generally cannot finish. Once the reduced
membership becomes pending, surviving workers preserve the last committed
cursor and enter the shared replacement path in Section 5.3. Any uncommitted
attempt is replayed after the new generation becomes active.

#### 5.2. Register added nodes and publish expanded membership

A new node starts one CPU agent and its configured GPU workers. The agent
registers a new stable node ID, incarnation, addresses, and GPU inventory with
the master. Registration proposes a complete membership generation containing
both incumbent and joining nodes; it does not add ranks to the active WORLD in
place.

```mermaid
sequenceDiagram
    participant JA as Joining node CPU agent
    participant JW as Joining node GPU workers
    participant M as CPU master process
    participant IA as Incumbent node agents
    participant IW as Incumbent GPU workers

    JA->>M: register(new stable ID, incarnation, addresses, GPU IDs)
    M->>M: validate registration and publish generation N+1
    M-->>JA: complete expanded membership snapshot N+1
    M-->>IA: complete expanded membership snapshot N+1
    JA-->>JW: snapshot N+1 over local IPC
    IA-->>IW: snapshot N+1 over local IPC
    par Build the expanded plan
        IW->>IW: apply_membership and compose the join plan
        JW->>JW: validate snapshot and compose the same plan
    end
    Note over JW,IW: Continue through the shared generation replacement in Section 5.3
```

When [`planning/reconfiguration.py`](oobleck/planning/reconfiguration.py) sees
a node identity that was not in the previous execution plan, it calls the
heterogeneous composer over the entire expanded membership, up to
`max_nodes`, and records the strategy as `join`. The objective remains
throughput first. If candidates have equal predicted iteration time, it
maximizes retained state bytes, minimizes moved incumbent nodes, and finally
uses stable identities as deterministic tie-breakers. Consequently, a join may
create another pipeline replica, enlarge existing pipelines, or choose a
different heterogeneous combination; it is not necessarily an append-only
layout change.

All workers derive a new stable rank map, so an incumbent numeric global rank
may change when a lexicographically earlier node ID joins even though stable
node and pipeline identities remain retained. Each worker validates the fixed
per-node TP width, compatibility digest, snapshot hash, and plan checksum before
the master publishes rendezvous data.

The current runtime uses the same pending-generation commit guard for joins and
failures. If an expanded membership becomes visible before the atomic step
commit, that attempt does not commit and is replayed after Section 5.3. If the
step commits first, its new state and sampler cursor become the recovery source,
and training resumes from the following logical batch.

If the expanded membership cannot form a supported template composition,
exceeds `max_nodes`, has the wrong TP width, or disagrees on compatibility or
plan checksums, the proposed generation cannot become active. A still newer
join, failure, drain, or replacement snapshot supersedes the proposal in the
same way as any other membership event.

#### 5.3. Replace the generation and transfer committed state

The generation-replacement methods layered onto `OobleckParallelContext` in
[`runtime.py`](oobleck/runtime.py) perform a common prepare phase followed by an
activate-and-recover phase. `apply_membership()` first updates the stable
membership and rank map and builds a pending execution plan. Removed nodes use
the simple, borrow, or merge planner; added nodes use the `join` composition
path. In both cases, the next plan is chosen before the active WORLD is retired.

```mermaid
flowchart TD
    EVENT["Receive complete membership snapshot"] --> PLAN["Build deterministic failure, drain,<br/>replacement, or join plan"]
    PLAN --> ISNAP
    PLAN --> JCOMPILE

    subgraph incumbents["On incumbent GPU workers"]
        ISNAP["Capture committed model, optimizer,<br/>scheduler, scaler, and cursor state"]
        ICLOSE["Invalidate prefetch and close schedules,<br/>meshes, partitions, and sync groups"]
        IDESTROY["Concurrently shut down groups,<br/>destroy WORLD, clear c10d registries"]
        ICOMPILE["Compile new rank-local ownership<br/>without WORLD"]
        ISNAP --> ICLOSE --> IDESTROY --> ICOMPILE
    end

    subgraph joiners["On joining GPU workers, if any"]
        JCOMPILE["Compile target ownership without WORLD;<br/>no pre-generation state is available"]
    end

    ICOMPILE --> PREP["All-agent prepared barrier"]
    JCOMPILE --> PREP
    PREP --> INIT["Create replacement WORLD<br/>and activate local partitions"]
    INIT --> MANIFEST["Gather old and new logical manifests;<br/>plan retained and missing bundles"]
    MANIFEST --> COPY["Variable-split all_to_all_single rounds<br/>from committed sources to target destinations"]
    COPY --> RESTORE["Restore tensors and training metadata;<br/>rebuild gradient groups"]
    RESTORE --> READY["All-agent ready barrier"]
    READY --> ACTIVE["Master broadcasts generation_active"]
    ACTIVE --> PENDING{"Was a logical batch left uncommitted?"}
    PENDING -- "Yes" --> REPLAY["Replay it from the unchanged committed cursor"]
    PENDING -- "No" --> NEXT["Continue with the next logical batch"]
```

**Prepare the new generation:**

1. On incumbent workers, [`recovery_base.py`](oobleck/recovery_base.py) captures
   the last committed parameter, persistent-buffer, optimizer, scheduler,
   scaler, and step state. Joining workers have no pre-generation snapshot.
2. Incumbent DataLoaders invalidate prefetched but uncommitted batches.
3. Incumbents close the active Cornstarch partition, schedules, meshes, and
   heterogeneous gradient groups. Joining workers have no old partition to
   retire.
4. [`distributed/lifecycle.py`](oobleck/distributed/lifecycle.py) concurrently
   shuts down every known backend on incumbents, destroys the old WORLD, clears
   the audited PyTorch c10d registries, and verifies that WORLD is no longer
   initialized.
5. Every participating worker compiles the target rank-local ownership while no
   distributed world exists and reports the plan through the prepared barrier.

**Activate and recover after rendezvous:**

1. Incumbent and joining workers create the replacement WORLD and activate their
   compiled local partitions.
2. [`state_base.py`](oobleck/state_base.py) matches old and new state through
   stable logical keys, TP lanes, placements, and committed versions.
3. [`state.py`](oobleck/state.py) plans retained shards and missing transfers. It
   chunks complete parameter and optimizer bundles and balances source,
   destination, round, and link load across available committed replicas.
4. [`state_transfer.py`](oobleck/state_transfer.py) executes the immutable,
   checksummed schedule with variable-split `all_to_all_single` rounds. Joining
   workers enter as destinations without a local snapshot; incumbents may be
   sources, destinations, or both under the new layout.
5. [`recovery.py`](oobleck/recovery.py) restores parameters, buffers, optimizer
   slots and groups, scheduler, scaler, and committed-step metadata, then
   rebuilds heterogeneous gradient synchronization.

After every participating worker reports ready, the master activates the
replacement generation. An interrupted logical batch is fetched again from the
unchanged committed cursor and can commit exactly once. If the previous batch
committed before the membership transition became pending, execution instead
continues from the next logical batch.
