# Pipeline planner and paper traceability

Oobleck owns offline profiling, pipeline-template generation, heterogeneous
composition, and elastic reconfiguration. The compiled Rust extension implements
the GPU--stage mapping cost model from Section 4.1.2 of the Oobleck paper. It does
not choose the template node specifications from Section 4.1.1 or compose
pipeline replicas from Section 4.2; those are separate planning layers.

## Refactored runtime specialization

The paper state is `T(S, u, v, d)`: layers `[u, v)` are partitioned into `S`
stages running on `d` GPUs. It also searches a GPU split `m` because several
stages may split the GPUs within one node.

The refactored Cornstarch runtime deliberately fixes tensor parallelism to one
complete node. A planner resource is therefore an indivisible fixed-TP logical
device, one resource is assigned to each stage, and:

```text
S = d = number of nodes in the template
```

This is the `S = n` endpoint of the paper's search. It preserves the paper's
layer-partition and timing algorithm, but does not search `S > n` or `m` because
the current runtime cannot instantiate variable-TP stages sharing one node.
`tensor_parallel_size` records the width of each logical device; it is not a
search dimension.

## Equations 1--4 in the implementation

For stage `i`, let `w_i = F_i + B_i`. The implementation selects the rightmost
slowest stage `k*`, matching the published artifact, and evaluates:

```text
T1 = sum(w_i)
T2 = (Nb - S + k* - 1) * w_k*
T3 = sum(w_i for i from k* through S - 1)
T  = T1 + T2 + T3
```

As specified in Section 4.1.2, template generation temporarily sets `Nb = 4S`.
Generated `PipelineTemplate` objects serialize `paper_t1`, `paper_t3`, and
`paper_bottleneck_stage`, so heterogeneous composition evaluates the same model
for its concrete microbatch allocation instead of switching to a minimax
surrogate.

## Exact optimized search

The old divide-and-conquer cache retained one locally best result for each
subproblem. That is unsafe: whether a subplan is best after composition depends
on `k*` and `T3`, not only its local total time.

The refactor evaluates the same paper objective without discarding that state.
For every possible bottleneck layer range `[a, b)` with latency `B`:

1. preceding stages may have latency at most `B`;
2. following stages must have latency strictly less than `B`, because `k*` is
   the rightmost bottleneck;
3. `T3` is the work of the complete layer suffix `[a, L)`; and
4. Equation 2 determines the cost for every feasible bottleneck position.

Because latency and memory are non-negative and additive, greedily taking the
longest feasible stage gives the minimum number of stages needed on either side
of a candidate bottleneck. Any larger feasible count is obtained by splitting
stages. The planner caches these bounds once, scans them for every requested
resource count, and reconstructs the lexicographically smallest equal-cost
partition.

Prefix sums make every stage latency and memory query constant time. Building
all bottleneck feasibility summaries takes `O(L^3)` time; selecting templates up
to `S` resources takes `O(S L^2)` additional candidate checks. The `O(L^2)`
stage table and bottleneck summaries are shared by every template generated in
the call.

## Memory and deterministic output

A stage is feasible only when its persistent state plus one microbatch of
activation memory fits the supplied device capacity. The final template records
the minimum per-stage microbatch capacity, and composition rejects larger batch
allocations.

Equal paper iteration times use the lexicographically smallest stage-start
vector. Rust and the Python oracle implement the same rule and return identical
serialized templates.

## Python/Rust API

`oobleck.planning.generator.create_pipeline_templates` is the supported API. It
validates the versioned profile, calls the Rust extension when installed, and
otherwise runs the equivalent Python implementation. Rust accepts only plain
profile records and scalar configuration; it does not import Cornstarch or
PyTorch distributed state. Invalid inputs and infeasible layouts become
`ValueError` through the supported API, while direct extension calls preserve a
separate `RuntimeError` for infeasible planning.
