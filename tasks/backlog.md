# Oobleck refactor backlog

These items are intentionally deferred until the DP/PP/TP elastic runtime and its recovery path are complete.

## Add context and expert parallelism

The first implementation supports LLM DP/PP/TP with fixed per-node TP width. Add Cornstarch context parallelism (CP) and expert parallelism (EP) after that path is stable.

This work must extend template resource modeling, heterogeneous mesh construction, logical state manifests, gradient synchronization, all-to-all state redistribution, replay determinism, and the single-GPU distributed tests. EP also needs expert ownership/migration policy; CP needs sequence-shard and dataloader semantics across a reconfiguration. Cover CP and EP independently before testing their composition with heterogeneous PP and elastic recovery.

## Generalize paper GPU--stage mapping beyond fixed per-node TP

The Rust planner now implements Section 4.1.2's `T1 + T2 + T3` objective for
the refactored runtime's supported `S = d = n` case: one complete fixed-TP node
per pipeline stage. Restore the paper's remaining `S > n` search and within-node
GPU split `m` only after Cornstarch can instantiate multiple pipeline stages in
one node with stage-specific TP widths. That work must extend template ownership,
rank-grid construction, profile data indexed by GPU width, heterogeneous meshes,
and state synchronization together; planner-only output that the runtime cannot
instantiate is not acceptable.
