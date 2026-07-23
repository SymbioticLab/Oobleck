# ADR 0003: Logical state identity and transactional batches

Parameter, buffer, and optimizer state is addressed by stable logical key, TP
lane, placement, and committed version. Missing bundles are chunked and assigned
largest-first by projected source/destination/link load, then transferred in a
checksummed all-to-all schedule. A logical global batch commits the optimizer,
scheduler, scaler, sampler cursor, and step together. Membership changes before
commit discard gradients and replay identical explicit sample indices.
