# ADR 0002: Replace the complete distributed universe

Every join, drain, replacement, or failure creates a monotonically newer full
membership snapshot. Workers close schedules and meshes, shut down known
backends concurrently, destroy WORLD, clear version-checked c10d registries,
compile ownership, create the replacement WORLD, activate local partitions, and
only then declare readiness. A newer snapshot supersedes partial recovery.
