"""Versioned profiles and a small, model-agnostic measurement backend."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from oobleck.types import CompatibilityFingerprint


PROFILE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class LayerExecutionResult:
    """Timing and memory measurements for one globally indexed model layer."""

    layer_index: int
    layer_name: str
    forward: float
    backward: float
    mem_required: int
    activation_memory: int = 0
    persistent_memory: int = 0

    def __post_init__(self) -> None:
        """Require a valid global identity and non-negative measurements."""

        if self.layer_index < 0 or not self.layer_name:
            raise ValueError("layer identity is invalid")
        if (
            not math.isfinite(self.forward)
            or not math.isfinite(self.backward)
            or not math.isfinite(self.forward + self.backward)
            or self.forward < 0
            or self.backward < 0
            or self.mem_required < 0
            or self.activation_memory < 0
            or self.persistent_memory < 0
        ):
            raise ValueError("profile measurements must be non-negative")


class JsonEncoder(json.JSONEncoder):
    """Encode profile value objects through their dataclass representation."""

    def default(self, obj: object) -> object:
        """Delegate unknown objects after handling profile dataclasses."""

        if isinstance(obj, (LayerExecutionResult, CompatibilityFingerprint)):
            return asdict(obj)
        return super().default(obj)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Versioned, compatibility-bound collection of per-layer measurements."""

    fingerprint: CompatibilityFingerprint
    microbatch_size: int
    layers: tuple[LayerExecutionResult, ...]
    schema_version: int = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject stale schemas and profiles without contiguous layer identities."""

        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema {self.schema_version}")
        if self.microbatch_size < 1 or not self.layers:
            raise ValueError("profile must contain a positive microbatch and layers")
        if tuple(item.layer_index for item in self.layers) != tuple(range(len(self.layers))):
            raise ValueError("profile layers must have contiguous global indices")

    def save(self, path: str | Path) -> None:
        """Write a deterministic JSON profile, creating parent directories."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), cls=JsonEncoder, sort_keys=True, indent=2) + "\n"
        )

    @classmethod
    def load(
        cls, path: str | Path, expected: CompatibilityFingerprint | None = None
    ) -> "ModelProfile":
        """Decode a profile and optionally enforce an exact fingerprint match."""

        target = Path(path)
        try:
            value = json.loads(target.read_text())
            profile = cls(
                CompatibilityFingerprint(**value["fingerprint"]),
                value["microbatch_size"],
                tuple(LayerExecutionResult(**item) for item in value["layers"]),
                value["schema_version"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load profile cache {target}: {exc}") from exc
        if expected is not None and profile.fingerprint != expected:
            raise ValueError(
                f"Profile cache {target} is stale or incompatible "
                f"(cached={profile.fingerprint.digest}, requested={expected.digest}). "
                "Regenerate it with the offline profile command."
            )
        return profile


ProfileInputFactory = Callable[[int, int], object]
ProfileLossFactory = Callable[[object], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ProfilingWorkload:
    """Materialized layers and input construction for offline profiling.

    ``input_factory(layer_index, iteration)`` may return one positional input,
    a tuple of positional inputs, or ``(args, kwargs)``. Supplying inputs per
    layer avoids imposing a Cornstarch model or pipeline signature here.
    """

    layers: tuple[torch.nn.Module, ...]
    input_factory: ProfileInputFactory
    loss_factory: ProfileLossFactory | None = None

    def __post_init__(self) -> None:
        """Require at least one layer and callable input/loss factories."""

        if not self.layers or not callable(self.input_factory):
            raise ValueError("a profiling workload requires layers and an input factory")
        if self.loss_factory is not None and not callable(self.loss_factory):
            raise ValueError("loss_factory must be callable")


class ModelProfiler:
    """Profile serialization plus a narrow forward/backward measurement API."""

    def __init__(
        self,
        tag: str,
        *,
        fingerprint: CompatibilityFingerprint | None = None,
        base_dir: str | Path = ".",
        **legacy: Any,
    ) -> None:
        """Bind cache layout and optional compatibility identity for this run."""

        self.tag = tag
        self.fingerprint = fingerprint
        self.profile_dir = Path(base_dir) / tag / "profile"
        self.legacy = legacy

    @staticmethod
    def get_profile_path(
        profile_dir: Path, tp_size: int, microbatch_size: int, precision: str
    ) -> Path:
        """Return the legacy cache path keyed by TP, batch size, and precision."""

        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir / f"profile_tp{tp_size}_mb{microbatch_size}_{precision}.json"

    def record(
        self,
        path: str | Path,
        microbatch_size: int,
        layers: Sequence[LayerExecutionResult],
    ) -> ModelProfile:
        """Validate, persist, and return a newly measured profile."""

        if self.fingerprint is None:
            raise ValueError("a compatibility fingerprint is required to record profiles")
        profile = ModelProfile(self.fingerprint, microbatch_size, tuple(layers))
        profile.save(path)
        return profile

    def load(self, path: str | Path) -> ModelProfile:
        """Load a profile under this profiler's compatibility requirement."""

        return ModelProfile.load(path, self.fingerprint)

    @staticmethod
    def _arguments(value: object) -> tuple[tuple[object, ...], dict[str, object]]:
        """Normalize flexible input-factory results into call args and kwargs."""

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            return value
        if isinstance(value, tuple):
            return value, {}
        if isinstance(value, dict):
            return (), value
        return (value,), {}

    @staticmethod
    def _default_loss(output: object) -> torch.Tensor:
        """Sum every differentiable floating output into a synthetic scalar loss."""

        tensors: list[torch.Tensor] = []

        def collect(value: object) -> None:
            """Recursively find floating tensor leaves in common containers."""

            if isinstance(value, torch.Tensor) and value.is_floating_point():
                tensors.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(output)
        differentiable = [item for item in tensors if item.requires_grad]
        if not differentiable:
            raise ValueError(
                "profiled layer output has no differentiable floating-point tensor; "
                "provide a loss_factory"
            )
        zero = torch.zeros((), device=differentiable[0].device)
        return sum((item.float().sum() for item in differentiable), zero)

    @staticmethod
    def _persistent_bytes(layer: torch.nn.Module) -> int:
        """Count parameter and buffer storage owned by a materialized layer."""

        state = tuple(layer.parameters(recurse=True)) + tuple(layer.buffers(recurse=True))
        return sum(item.numel() * item.element_size() for item in state)

    def measure(
        self,
        workload: ProfilingWorkload,
        *,
        warmup_steps: int = 2,
        measurement_steps: int = 5,
    ) -> tuple[LayerExecutionResult, ...]:
        """Measure materialized layers without legacy model wrappers.

        CUDA synchronization brackets each timing interval. ``mem_required``
        includes persistent state and peak additional CUDA allocation. CPU
        measurements report persistent bytes and support offline smoke tests.
        """

        if warmup_steps < 0 or measurement_steps < 1:
            raise ValueError("warmup_steps must be non-negative and measurement_steps positive")
        loss_factory = workload.loss_factory or self._default_loss
        results: list[LayerExecutionResult] = []
        for layer_index, layer in enumerate(workload.layers):
            try:
                first_tensor = next(layer.parameters())
            except StopIteration:
                first_tensor = next(layer.buffers(), None)
            device = (
                first_tensor.device
                if first_tensor is not None and first_tensor.device.type == "cuda"
                else None
            )
            if device is not None:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline_memory = torch.cuda.memory_allocated(device)
            else:
                device = None
                baseline_memory = 0

            forward_seconds = 0.0
            backward_seconds = 0.0
            total_steps = warmup_steps + measurement_steps
            for iteration in range(total_steps):
                layer.zero_grad(set_to_none=True)
                args, kwargs = self._arguments(workload.input_factory(layer_index, iteration))
                if device is not None:
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = layer(*args, **kwargs)
                if device is not None:
                    torch.cuda.synchronize(device)
                after_forward = time.perf_counter()
                loss = loss_factory(output)
                loss.backward()
                if device is not None:
                    torch.cuda.synchronize(device)
                after_backward = time.perf_counter()
                if iteration >= warmup_steps:
                    forward_seconds += after_forward - started
                    backward_seconds += after_backward - after_forward

            peak_extra = 0
            if device is not None:
                peak_extra = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
            layer_name = getattr(layer, "_oobleck_profile_name", None)
            persistent_memory = self._persistent_bytes(layer)
            results.append(
                LayerExecutionResult(
                    layer_index,
                    str(layer_name or f"{type(layer).__name__}.{layer_index}"),
                    forward_seconds / measurement_steps,
                    backward_seconds / measurement_steps,
                    persistent_memory + peak_extra,
                    peak_extra,
                    persistent_memory,
                )
            )
        return tuple(results)

    def measure_and_record(
        self,
        path: str | Path,
        microbatch_size: int,
        workload: ProfilingWorkload,
        *,
        warmup_steps: int = 2,
        measurement_steps: int = 5,
    ) -> ModelProfile:
        """Measure a workload and atomically feed the results into profile storage."""

        return self.record(
            path,
            microbatch_size,
            self.measure(
                workload,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
            ),
        )

    def load_profile(self, microbatch_size: int) -> list[LayerExecutionResult]:
        """Load layers from the retained legacy cache naming convention."""

        precision = str(self.legacy.get("precision", "unknown"))
        tp_size = int(self.legacy.get("tp_size", 1))
        path = self.get_profile_path(self.profile_dir, tp_size, microbatch_size, precision)
        return list(ModelProfile.load(path, self.fingerprint).layers)
