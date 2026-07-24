# Pipeline planner boundary

Oobleck owns offline profiling, pipeline-template generation, heterogeneous
composition, and elastic reconfiguration. The compiled Rust extension is a
narrow accelerator for **template partitioning only**; it does not choose
membership, compose replicas, allocate the global batch, or initialize a
distributed runtime.

## Supported planning model

The refactor release uses a fixed tensor-parallel width per node. One template
resource corresponds to one pipeline stage backed by one node, so a template
for `n` resources contains exactly `n` non-empty contiguous global layer
ranges. This is intentionally narrower than the variable GPU-to-stage mapping
in Section 4.1.2 of the Oobleck paper.

For every requested resource count, both the Rust backend and Python fallback
minimize the maximum per-stage `forward + backward` profile time. This minimax
objective is the steady-state coefficient used by `PipelineTemplate.iteration_time`.
Equal-cost layouts use the lexicographically smallest stage-start vector, which
makes generated ownership deterministic across processes and backends.

If device memory is supplied, a candidate stage is feasible only when its
persistent state plus one microbatch of activation memory fits. The generated
template records the minimum per-stage microbatch capacity. Composition later
honors that capacity when assigning the fixed global batch.

## Python/Rust API

`oobleck.planning.generator.create_pipeline_templates` is the supported API.
It validates the versioned Python profile, calls
`oobleck.planning.planner.create_pipeline_templates` when the extension is
installed, and otherwise executes the equivalent Python dynamic program. Both
backends return the same serialized `PipelineTemplate` fields and layer ranges.

The Rust extension accepts only plain profile records and scalar configuration;
it never imports Cornstarch, PyTorch distributed state, or legacy ColossalAI
objects. Invalid inputs and infeasible layouts raise `ValueError` from the supported
Python API. Direct extension callers receive `RuntimeError` for an infeasible
layout, preserving a distinct native error category.

## Algorithm and complexity

Prefix sums make every contiguous stage metric an O(1) lookup. The dynamic
program stores the best partition for each `(stage_count, prefix_length)` and
examines every final cut, giving O(S L²) candidate evaluations for `L` layers
and at most `S` stages. Backpointers are represented by the stable stage-start
vector because template generation is offline and model layer counts are
small; this keeps tie-breaking auditable without concurrent hash maps or
schedule-dependent reductions.
