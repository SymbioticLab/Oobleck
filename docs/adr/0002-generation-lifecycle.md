# ADR 0002: Replace the complete distributed universe

Every join, drain, replacement, or failure creates a monotonically newer full
membership snapshot. Workers close schedules and meshes, shut down known
backends concurrently, destroy WORLD, clear version-checked c10d registries,
and compile ownership without a distributed world. Each worker then reports the
checksummed plan as prepared. Only after every agent agrees does the master
publish rendezvous parameters; workers create the replacement WORLD, activate
local partitions, recover state, and report readiness. The master marks the
generation active only after that final barrier. A newer membership snapshot
supersedes any prepared or partially activated recovery.
