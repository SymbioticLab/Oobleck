"""Runtime and dataset compatibility fingerprints for elastic generation consensus."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from oobleck.types import RuntimeCompatibility, checksum


def _package_revision(distribution: str, module_name: str) -> str:
    """Find the strongest available package identity: VCS commit, then version."""

    try:
        direct_url = importlib.metadata.distribution(distribution).read_text("direct_url.json")
        if direct_url:
            value = json.loads(direct_url)
            commit = value.get("vcs_info", {}).get("commit_id")
            if isinstance(commit, str) and commit:
                return commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    try:
        module = importlib.import_module(module_name)
        location = Path(module.__file__).resolve()
        for parent in location.parents:
            if (parent / ".git").exists():
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    timeout=2,
                ).strip()
    except (ImportError, OSError, subprocess.SubprocessError):
        pass
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def model_fingerprint(model: torch.nn.Module, *, model_identity: str = "structure-only") -> str:
    """Hash model type and state layout without reading mutable tensor contents."""

    if not model_identity:
        raise ValueError("model_identity must not be empty")
    state = [
        (kind, name, tuple(value.shape), str(value.dtype))
        for kind, values in (
            ("parameter", model.named_parameters(recurse=True, remove_duplicate=False)),
            ("buffer", model.named_buffers(recurse=True, remove_duplicate=False)),
        )
        for name, value in values
    ]
    return checksum(
        {
            "type": f"{type(model).__module__}.{type(model).__qualname__}",
            "identity": model_identity,
            "state": state,
        }
    )


def dataset_fingerprint(
    dataset: object,
    *,
    explicit: str | None = None,
    preprocessing: str = "identity",
) -> str:
    """Hash a stable indexed dataset identity together with preprocessing semantics.

    The fingerprint deliberately rejects iterable datasets and anonymous
    map-style datasets: generation consensus must not equate workers whose
    sample ordering or preprocessing may differ.
    """

    if isinstance(dataset, IterableDataset) or not (
        hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
    ):
        raise TypeError("runtime compatibility requires a stable map-style dataset")
    length = len(dataset)  # type: ignore[arg-type]
    if length < 1:
        raise ValueError("runtime compatibility requires a non-empty dataset")
    intrinsic = explicit
    if intrinsic is None:
        intrinsic = getattr(dataset, "_fingerprint", None)
    if intrinsic is None:
        intrinsic = getattr(dataset, "oobleck_fingerprint", None)
    if not isinstance(intrinsic, str) or not intrinsic:
        raise ValueError("dataset has no stable fingerprint; pass dataset_fingerprint explicitly")
    if not preprocessing:
        raise ValueError("preprocessing fingerprint must not be empty")
    return checksum(
        {
            "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "length": length,
            "intrinsic": intrinsic,
            "preprocessing": preprocessing,
        }
    )


def _hardware_fingerprint() -> str:
    """Hash the local accelerator model or CPU platform used for this worker."""

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        value: dict[str, Any] = {
            "kind": "cuda",
            "name": properties.name,
            "capability": torch.cuda.get_device_capability(device),
            "total_memory": properties.total_memory,
        }
    else:
        value = {
            "kind": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    return checksum(value)


def build_runtime_compatibility(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_identity: str,
    dataset_identity: str | None = None,
    preprocessing_fingerprint: str = "identity",
    oobleck_revision: str | None = None,
    cornstarch_revision: str | None = None,
) -> RuntimeCompatibility:
    """Assemble the exact software, model, dataset, and hardware generation contract.

    Package revisions prefer source commits, while Torch/CUDA/NCCL and datasets versions describe
    the executable environment. Model structure, stable dataset/preprocessing identity, schema
    versions, and local hardware are fingerprinted independently. The resulting immutable digest
    is embedded in execution plans and must agree across workers before rendezvous.
    """

    cuda_version = torch.version.cuda
    nccl_version = None
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        try:
            nccl_version = ".".join(str(item) for item in torch.cuda.nccl.version())
        except (AttributeError, RuntimeError, TypeError):
            nccl_version = "available"
    try:
        datasets_version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError:
        datasets_version = "not-installed"
    return RuntimeCompatibility(
        oobleck_revision or _package_revision("oobleck", "oobleck"),
        cornstarch_revision or _package_revision("cornstarch", "cornstarch"),
        torch.__version__,
        cuda_version,
        nccl_version,
        model_fingerprint(model, model_identity=model_identity),
        1,
        1,
        datasets_version,
        dataset_fingerprint(
            dataset,
            explicit=dataset_identity,
            preprocessing=preprocessing_fingerprint,
        ),
        _hardware_fingerprint(),
    )


__all__ = [
    "build_runtime_compatibility",
    "dataset_fingerprint",
    "model_fingerprint",
]
