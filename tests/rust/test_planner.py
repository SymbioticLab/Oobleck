import pytest
from oobleck import planner
from pathlib import Path
import csv

model_name = "gpt2"
tag = "test"


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    path = tmp_path / "profiles" / f"{model_name}__{tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        fieldnames = [
            "layer_index",
            "layer_name",
            "forward",
            "backward",
            "mem_required",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(6):
            writer.writerow(
                {
                    "layer_index": i,
                    "layer_name": f"layer_{i}",
                    "forward": i + 1,
                    "backward": 1 + 1,
                    "mem_required": i + 1,
                }
            )

    return tmp_path


def test_error_for_too_large_num_nodes(base_dir: Path):
    with pytest.raises(RuntimeError):
        planner.create_pipeline_templates(
            model_name="gpt2", tag="test", num_nodes=[8], oobleck_base_dir=base_dir
        )


def test_create_pipeline_templates(base_dir: Path):
    template_layers: dict[int, list[list[str]]] = planner.create_pipeline_templates(
        model_name="gpt2", tag="test", num_nodes=[1, 2, 3, 4], oobleck_base_dir=base_dir
    )

    expected_layers = [f"layer_{i}" for i in range(6)]

    assert sorted(list(template_layers.keys())) == [1, 2, 3, 4]
    for _, template in template_layers.items():
        covered_layers = []
        for stage in template:
            for layer in stage:
                covered_layers.append(layer)

        assert expected_layers == covered_layers
