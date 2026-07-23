"""Versioned profile/template cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from oobleck.types import CompatibilityFingerprint, PipelineTemplate


TEMPLATE_CACHE_SCHEMA = 1


def save_templates(
    path: str | Path,
    templates: Sequence[PipelineTemplate],
    fingerprint: CompatibilityFingerprint,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TEMPLATE_CACHE_SCHEMA,
        "fingerprint": fingerprint.__dict__
        if hasattr(fingerprint, "__dict__")
        else {
            "model": fingerprint.model,
            "dtype": fingerprint.dtype,
            "tensor_parallel_size": fingerprint.tensor_parallel_size,
            "hardware": fingerprint.hardware,
            "cornstarch_version": fingerprint.cornstarch_version,
            "schema_version": fingerprint.schema_version,
        },
        "templates": [item.to_dict() for item in templates],
    }
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def load_templates(
    path: str | Path, fingerprint: CompatibilityFingerprint
) -> tuple[PipelineTemplate, ...]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read template cache {target}: {exc}") from exc
    if payload.get("schema_version") != TEMPLATE_CACHE_SCHEMA:
        raise ValueError(
            f"Template cache {target} uses schema {payload.get('schema_version')!r}; "
            f"expected {TEMPLATE_CACHE_SCHEMA}. Regenerate it."
        )
    cached = CompatibilityFingerprint(**payload["fingerprint"])
    if cached != fingerprint:
        raise ValueError(
            f"Template cache {target} is incompatible "
            f"(cached={cached.digest}, requested={fingerprint.digest}). Regenerate it."
        )
    templates = tuple(PipelineTemplate.from_dict(item) for item in payload["templates"])
    for template in templates:
        template.assert_compatible(fingerprint)
    return templates
