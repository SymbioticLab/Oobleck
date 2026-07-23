# Oobleck refactor backlog

These items are intentionally deferred until the DP/PP/TP elastic runtime and its recovery path are complete.

## Refactor the Rust pipeline planner

The initial Oobleck refactor will reuse the current Rust pipeline-template planner behind a narrow, versioned Python API. It will receive correctness fixes, tests, and the required PyO3 upgrade, but its internal design will not be substantially rewritten in the first implementation.

Later work should simplify the planner's data model, separate enumeration from optimization, replace legacy assumptions and naming, improve error reporting, benchmark a pure-Python or alternative solver where useful, and document the Rust/Python boundary. Preserve behavioral fixtures for template generation and simple/borrow/merge decisions before changing the implementation.

## Add context and expert parallelism

The first implementation supports LLM DP/PP/TP with fixed per-node TP width. Add Cornstarch context parallelism (CP) and expert parallelism (EP) after that path is stable.

This work must extend template resource modeling, heterogeneous mesh construction, logical state manifests, gradient synchronization, all-to-all state redistribution, replay determinism, and the single-GPU distributed tests. EP also needs expert ownership/migration policy; CP needs sequence-shard and dataloader semantics across a reconfiguration. Cover CP and EP independently before testing their composition with heterogeneous PP and elastic recovery.
