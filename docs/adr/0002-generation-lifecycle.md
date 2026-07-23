# ADR 0002: Replace the complete distributed universe

Every membership change creates a monotonically newer complete snapshot and a
replacement WORLD; Oobleck never edits the active process-group universe in
place. Workers close schedules and meshes, shut down known backends concurrently,
destroy WORLD, clear version-checked c10d registries, compile ownership without a
distributed world, recover committed state, and pass the ready barrier before the
master marks the generation active.

## Cutover decision

A proposal is a **graceful addition** only when its accumulated operation set contains at least one added incarnation and no removed incarnations. If no step is running, preparation
starts immediately. If a step is running, that one step may commit under the old
generation; the runtime then promotes the newest coalesced addition plan and blocks
the next step until `generation_active`. The committed batch is not replayed.

Any snapshot containing a removal is a **hard transition**, including a same-ID restart or an addition that supersedes removal recovery. They immediately make the active attempt uncommittable. A hard
transition also supersedes any deferred addition, and the interrupted logical batch
is replayed after recovery. A newer snapshot always supersedes an older prepared
or partially activated generation.

## Plan and state consensus

The control protocol carries the complete versioned, checksummed execution
plan in prepared acknowledgements. The master retains it as the previous active
plan only after readiness consensus activates that generation, then binds it into
the hash of later membership proposals. Fresh workers seed reconfiguration from
that plan, configure their DataLoader and optimization objects before recovery,
and receive model, optimizer, scheduler, scaler, committed-step, sampler-epoch,
and sampler-cursor state from incumbents.
