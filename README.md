# Oobleck

Oobleck is a heterogeneous elastic pipeline-training runtime built around
Cornstarch's plan/context model. It chooses rank-local ownership before WORLD,
uses checksummed membership generations, fully replaces the process-group
universe on every change, and treats each logical global batch as a replayable
transaction.

```python
plan = OobleckParallelizationPlan(
    OobleckConfig(global_batch_size=128, microbatch_size=2, max_nodes=16)
)
plan.parallelize(model, ParallelConfig(tensor_parallel_size=8))
context = plan.materialize("cuda", dtype=torch.bfloat16)

sampler = context.create_batch_sampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, persistent_workers=False)
loader = context.prepare_dataloader(dataloader)
context.configure_optimization(
    optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=1e-4)
)
for batch in loader:
    context.step(batch, execution_plan, output, criterion)
```

The first release supports stable map-style datasets and fixed per-node tensor
parallel width. Oobleck selects heterogeneous pipeline and effective data
parallel layouts. Context/expert parallelism, streaming-dataset replay, elastic
TP width, multimodal models, and master high availability are out of scope.

See [the examples](examples/README.md), the architecture decisions in
`docs/adr/`, and `tasks/plan.md` for the full design and acceptance contract.

## Development

```bash
pytest -q
cargo test
python examples/run_local.py
python benchmarks/recovery_schedule.py
python benchmarks/transaction_overhead.py
```

The Rust planner binding requires PyO3 and `pyo3-build-config` 0.24.1 or newer;
its exact transitive versions are recorded in `Cargo.lock`.
