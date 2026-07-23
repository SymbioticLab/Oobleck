import torch

from oobleck.optimization import restore_optimizer_state, serialize_optimizer_state
from oobleck.state import LogicalStateEntry


def test_adam_state_round_trips_by_logical_parameter_key():
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    source(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    schema, tensors = serialize_optimizer_state(
        optimizer,
        dict(source.named_parameters()),
        owner_rank=0,
        committed_step=1,
    )

    destination = torch.nn.Linear(3, 2)
    restored = torch.optim.AdamW(destination.parameters(), lr=0.5)
    restore_optimizer_state(restored, schema, tensors, dict(destination.named_parameters()))
    assert restored.param_groups[0]["lr"] == 0.01
    for source_parameter, destination_parameter in zip(
        source.parameters(), destination.parameters()
    ):
        for slot in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[source_parameter][slot],
                restored.state[destination_parameter][slot],
            )


def test_optimizer_tensor_entries_inherit_tp_shard_identity():
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    parameter_entry = LogicalStateEntry(
        "weight",
        (4, 3),
        (2, 3),
        "torch.float32",
        "parameter",
        ("Shard(dim=0)",),
        2,
        7,
        4,
    )
    schema, _ = serialize_optimizer_state(
        optimizer,
        dict(model.named_parameters()),
        owner_rank=7,
        committed_step=4,
        parameter_entries={"weight": parameter_entry},
    )
    shaped = [entry for entry in schema.tensor_entries if entry.local_shape == (2, 3)]
    assert shaped
    assert all(entry.tp_lane == 2 for entry in schema.tensor_entries)
    assert shaped[0].global_shape == (4, 3)
    assert shaped[0].placements == ("Shard(dim=0)",)
