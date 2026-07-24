"""Machine-readable Rust/Python pipeline planner benchmark."""

from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from oobleck.planning import LayerExecutionResult, create_pipeline_templates
from oobleck.planning.generator import _create_python_templates


def run(*, layers: int = 96, max_resources: int = 64, repeats: int = 3) -> dict[str, object]:
    if layers < 1 or max_resources < 1 or max_resources > layers or repeats < 1:
        raise ValueError("layers, resources, and repeats must be positive with resources <= layers")
    profile = tuple(
        LayerExecutionResult(
            index,
            f"layer.{index}",
            (index % 13 + 1) / 1000,
            (index % 7 + 1) / 1000,
            1024,
            768,
            256,
        )
        for index in range(layers)
    )
    resource_counts = tuple(range(1, max_resources + 1))
    rust_seconds = []
    python_seconds = []
    rust_templates = {}
    python_templates = {}
    for _ in range(repeats):
        started = time.perf_counter()
        rust_templates = create_pipeline_templates("benchmark", profile, resource_counts)
        rust_seconds.append(time.perf_counter() - started)

        started = time.perf_counter()
        python_templates = _create_python_templates(
            "benchmark", profile, resource_counts, 1, None, None
        )
        python_seconds.append(time.perf_counter() - started)

    rust_average = mean(rust_seconds)
    python_average = mean(python_seconds)
    return {
        "schema_version": 1,
        "scenario": {
            "layers": layers,
            "max_resources": max_resources,
            "template_count": len(resource_counts),
            "repeats": repeats,
        },
        "backends_match": rust_templates == python_templates,
        "rust_seconds": rust_average,
        "python_seconds": python_average,
        "rust_speedup": python_average / rust_average if rust_average else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument("--max-resources", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                layers=arguments.layers,
                max_resources=arguments.max_resources,
                repeats=arguments.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
