# Oobleck refactor backlog

These items are intentionally deferred until the DP/PP/TP elastic runtime and its recovery path are complete.

## Add context and expert parallelism

The first implementation supports LLM DP/PP/TP with fixed per-node TP width. Add Cornstarch context parallelism (CP) and expert parallelism (EP) after that path is stable.

This work must extend template resource modeling, heterogeneous mesh construction, logical state manifests, gradient synchronization, all-to-all state redistribution, replay determinism, and the single-GPU distributed tests. EP also needs expert ownership/migration policy; CP needs sequence-shard and dataloader semantics across a reconfiguration. Cover CP and EP independently before testing their composition with heterogeneous PP and elastic recovery.
