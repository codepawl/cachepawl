"""Result schema for one benchmark run.

JSON round-tripable via ``BenchmarkRun.to_json`` and
``BenchmarkRun.from_json``. The schema is versioned; consumers that load
an older artifact should branch on ``schema_version``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, cast

from cachepawl.benchmarks.harness.metrics import (
    AllocatorMetrics,
    LatencyPercentiles,
    compute_percentiles,
)
from cachepawl.benchmarks.harness.workloads import (
    AttentionLayerProfile,
    SSMLayerProfile,
    WorkloadSpec,
)
from cachepawl.quant.dtypes import DType

SCHEMA_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True, slots=True)
class Hardware:
    """Hardware fingerprint captured for the run."""

    device: str
    gpu_name: str | None
    vram_total_bytes: int | None
    cuda_capability: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class Environment:
    """Software fingerprint captured for the run."""

    torch_version: str
    numpy_version: str
    cachepawl_version: str
    cuda_version: str | None
    python_version: str


@dataclass(slots=True)
class BenchmarkRun:
    """One end-to-end benchmark run plus its provenance."""

    spec: WorkloadSpec
    allocator_name: str
    hardware: Hardware
    environment: Environment
    started_at: str
    finished_at: str
    metrics: AllocatorMetrics
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec": _spec_to_dict(self.spec),
            "allocator_name": self.allocator_name,
            "hardware": _hardware_to_dict(self.hardware),
            "environment": _environment_to_dict(self.environment),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": _metrics_to_dict(self.metrics),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BenchmarkRun:
        return cls(
            schema_version=_pop_str(data, "schema_version"),
            spec=_spec_from_dict(_pop_dict(data, "spec")),
            allocator_name=_pop_str(data, "allocator_name"),
            hardware=_hardware_from_dict(_pop_dict(data, "hardware")),
            environment=_environment_from_dict(_pop_dict(data, "environment")),
            started_at=_pop_str(data, "started_at"),
            finished_at=_pop_str(data, "finished_at"),
            metrics=_metrics_from_dict(_pop_dict(data, "metrics")),
            notes=_pop_str(data, "notes"),
        )

    @classmethod
    def from_json(cls, text: str) -> BenchmarkRun:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("BenchmarkRun JSON root must be an object")
        return cls.from_dict(cast(dict[str, object], parsed))


def _spec_to_dict(spec: WorkloadSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "num_requests": spec.num_requests,
        "attention_layers": spec.attention_layers,
        "ssm_layers": spec.ssm_layers,
        "attention_profile": {
            "num_kv_heads": spec.attention_profile.num_kv_heads,
            "head_dim": spec.attention_profile.head_dim,
        },
        "ssm_profile": {
            "d_inner": spec.ssm_profile.d_inner,
            "d_state": spec.ssm_profile.d_state,
        },
        "dtype": spec.dtype.value,
        "seed": spec.seed,
    }


def _spec_from_dict(data: dict[str, object]) -> WorkloadSpec:
    attn_raw = _pop_dict(data, "attention_profile")
    ssm_raw = _pop_dict(data, "ssm_profile")
    return WorkloadSpec(
        name=_pop_str(data, "name"),
        num_requests=_pop_int(data, "num_requests"),
        attention_layers=_pop_int(data, "attention_layers"),
        ssm_layers=_pop_int(data, "ssm_layers"),
        attention_profile=AttentionLayerProfile(
            num_kv_heads=_pop_int(attn_raw, "num_kv_heads"),
            head_dim=_pop_int(attn_raw, "head_dim"),
        ),
        ssm_profile=SSMLayerProfile(
            d_inner=_pop_int(ssm_raw, "d_inner"),
            d_state=_pop_int(ssm_raw, "d_state"),
        ),
        dtype=DType(_pop_str(data, "dtype")),
        seed=_pop_int(data, "seed"),
    )


def _hardware_to_dict(hw: Hardware) -> dict[str, object]:
    return {
        "device": hw.device,
        "gpu_name": hw.gpu_name,
        "vram_total_bytes": hw.vram_total_bytes,
        "cuda_capability": (list(hw.cuda_capability) if hw.cuda_capability is not None else None),
    }


def _hardware_from_dict(data: dict[str, object]) -> Hardware:
    cap_raw = data.get("cuda_capability")
    cap: tuple[int, int] | None
    if cap_raw is None:
        cap = None
    elif isinstance(cap_raw, list) and len(cap_raw) == 2:
        first, second = cap_raw
        cap = (int(cast(int, first)), int(cast(int, second)))
    else:
        raise ValueError(f"cuda_capability must be a 2-element list, got {cap_raw!r}")
    return Hardware(
        device=_pop_str(data, "device"),
        gpu_name=_pop_optional_str(data, "gpu_name"),
        vram_total_bytes=_pop_optional_int(data, "vram_total_bytes"),
        cuda_capability=cap,
    )


def _environment_to_dict(env: Environment) -> dict[str, object]:
    return {
        "torch_version": env.torch_version,
        "numpy_version": env.numpy_version,
        "cachepawl_version": env.cachepawl_version,
        "cuda_version": env.cuda_version,
        "python_version": env.python_version,
    }


def _environment_from_dict(data: dict[str, object]) -> Environment:
    return Environment(
        torch_version=_pop_str(data, "torch_version"),
        numpy_version=_pop_str(data, "numpy_version"),
        cachepawl_version=_pop_str(data, "cachepawl_version"),
        cuda_version=_pop_optional_str(data, "cuda_version"),
        python_version=_pop_str(data, "python_version"),
    )


def _metrics_to_dict(m: AllocatorMetrics) -> dict[str, object]:
    allocate_p = compute_percentiles(m.allocate_latency_ns)
    free_p = compute_percentiles(m.free_latency_ns)
    return {
        "peak_reserved_bytes": m.peak_reserved_bytes,
        "peak_allocated_bytes": m.peak_allocated_bytes,
        "fragmentation_samples": list(m.fragmentation_samples),
        "allocate_latency_ns": list(m.allocate_latency_ns),
        "free_latency_ns": list(m.free_latency_ns),
        "allocate_latency_percentiles": _percentiles_to_dict(allocate_p),
        "free_latency_percentiles": _percentiles_to_dict(free_p),
        "oom_count": m.oom_count,
        "preemption_count": m.preemption_count,
        "active_requests_samples": list(m.active_requests_samples),
    }


def _metrics_from_dict(data: dict[str, object]) -> AllocatorMetrics:
    return AllocatorMetrics(
        peak_reserved_bytes=_pop_int(data, "peak_reserved_bytes"),
        peak_allocated_bytes=_pop_int(data, "peak_allocated_bytes"),
        fragmentation_samples=_pop_float_list(data, "fragmentation_samples"),
        allocate_latency_ns=_pop_int_list(data, "allocate_latency_ns"),
        free_latency_ns=_pop_int_list(data, "free_latency_ns"),
        oom_count=_pop_int(data, "oom_count"),
        preemption_count=_pop_int(data, "preemption_count"),
        active_requests_samples=_pop_int_list(data, "active_requests_samples"),
    )


def _percentiles_to_dict(p: LatencyPercentiles) -> dict[str, int]:
    return {
        "p50_ns": p.p50_ns,
        "p95_ns": p.p95_ns,
        "p99_ns": p.p99_ns,
        "max_ns": p.max_ns,
    }


def _pop_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"expected string at key {key!r}, got {type(value).__name__}")
    return value


def _pop_optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected string or null at key {key!r}, got {type(value).__name__}")
    return value


def _pop_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected int at key {key!r}, got {type(value).__name__}")
    return value


def _pop_optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected int or null at key {key!r}, got {type(value).__name__}")
    return value


def _pop_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected object at key {key!r}, got {type(value).__name__}")
    return cast(dict[str, object], value)


def _pop_int_list(data: dict[str, object], key: str) -> list[int]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"expected list at key {key!r}, got {type(value).__name__}")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"expected int in list {key!r}, got {type(item).__name__}")
        out.append(item)
    return out


def _pop_float_list(data: dict[str, object], key: str) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"expected list at key {key!r}, got {type(value).__name__}")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"expected float in list {key!r}, got bool")
        if not isinstance(item, (int, float)):
            raise ValueError(f"expected float in list {key!r}, got {type(item).__name__}")
        out.append(float(item))
    return out


__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkRun",
    "Environment",
    "Hardware",
]
