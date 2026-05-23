"""JSONL schema for cache-planning benchmark probe records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final, TypeAlias, cast

BENCH_RESULT_SCHEMA_VERSION: Final[str] = "0.1.0"
MetadataValue: TypeAlias = str | int | float | bool | None
Metadata: TypeAlias = dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class GpuMetadata:
    """GPU/runtime metadata captured without requiring CUDA."""

    name: str | None
    total_memory_bytes: int | None
    compute_capability: tuple[int, int] | None
    cuda_available: bool
    device_count: int

    def __post_init__(self) -> None:
        if self.total_memory_bytes is not None and self.total_memory_bytes < 0:
            raise ValueError("gpu.total_memory_bytes must be non-negative when provided")
        if self.device_count < 0:
            raise ValueError("gpu.device_count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_capability": (
                list(self.compute_capability) if self.compute_capability is not None else None
            ),
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GpuMetadata:
        capability_raw = data.get("compute_capability")
        capability: tuple[int, int] | None
        if capability_raw is None:
            capability = None
        elif isinstance(capability_raw, list) and len(capability_raw) == 2:
            first, second = capability_raw
            capability = (
                _as_int(first, "gpu.compute_capability[0]"),
                _as_int(second, "gpu.compute_capability[1]"),
            )
        else:
            raise ValueError("gpu.compute_capability must be null or a two-element list")
        return cls(
            name=_optional_str(data.get("name"), "gpu.name"),
            total_memory_bytes=_optional_int(
                data.get("total_memory_bytes"), "gpu.total_memory_bytes"
            ),
            compute_capability=capability,
            cuda_available=_as_bool(data.get("cuda_available"), "gpu.cuda_available"),
            device_count=_as_int(data.get("device_count"), "gpu.device_count"),
        )


@dataclass(frozen=True, slots=True)
class CacheProbeResult:
    """One cache-planning and memory-efficiency benchmark record."""

    run_id: str
    timestamp: str
    backend: str
    workload: str
    model: str
    gpu: GpuMetadata
    estimated_bytes: int
    reserved_bytes: int
    useful_bytes: int
    overestimation_ratio: float
    wasted_fraction: float
    virtual_oom: bool
    planner_runtime_us: float
    metadata: Metadata = field(default_factory=dict)
    schema_version: str = BENCH_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.timestamp, "timestamp")
        _require_non_empty(self.backend, "backend")
        _require_non_empty(self.workload, "workload")
        _require_non_empty(self.model, "model")
        _require_non_negative(self.estimated_bytes, "estimated_bytes")
        _require_non_negative(self.reserved_bytes, "reserved_bytes")
        _require_non_negative(self.useful_bytes, "useful_bytes")
        if self.useful_bytes == 0 and self.estimated_bytes > 0:
            raise ValueError("useful_bytes must be positive when estimated_bytes is positive")
        if self.overestimation_ratio < 0.0:
            raise ValueError("overestimation_ratio must be non-negative")
        if not 0.0 <= self.wasted_fraction <= 1.0:
            raise ValueError("wasted_fraction must be in [0.0, 1.0]")
        if self.planner_runtime_us < 0.0:
            raise ValueError("planner_runtime_us must be non-negative")
        for key, value in self.metadata.items():
            _require_non_empty(key, "metadata key")
            _validate_metadata_value(value, f"metadata[{key!r}]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "backend": self.backend,
            "workload": self.workload,
            "model": self.model,
            "gpu": self.gpu.to_dict(),
            "estimated_bytes": self.estimated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "useful_bytes": self.useful_bytes,
            "overestimation_ratio": self.overestimation_ratio,
            "wasted_fraction": self.wasted_fraction,
            "virtual_oom": self.virtual_oom,
            "planner_runtime_us": self.planner_runtime_us,
            "metadata": dict(self.metadata),
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CacheProbeResult:
        version = _pop_str(data, "schema_version")
        if version != BENCH_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r}; expected {BENCH_RESULT_SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=version,
            run_id=_pop_str(data, "run_id"),
            timestamp=_pop_str(data, "timestamp"),
            backend=_pop_str(data, "backend"),
            workload=_pop_str(data, "workload"),
            model=_pop_str(data, "model"),
            gpu=GpuMetadata.from_dict(_pop_dict(data, "gpu")),
            estimated_bytes=_pop_int(data, "estimated_bytes"),
            reserved_bytes=_pop_int(data, "reserved_bytes"),
            useful_bytes=_pop_int(data, "useful_bytes"),
            overestimation_ratio=_pop_float(data, "overestimation_ratio"),
            wasted_fraction=_pop_float(data, "wasted_fraction"),
            virtual_oom=_pop_bool(data, "virtual_oom"),
            planner_runtime_us=_pop_float(data, "planner_runtime_us"),
            metadata=_pop_metadata(data, "metadata"),
        )

    @classmethod
    def from_json_line(cls, text: str) -> CacheProbeResult:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("CacheProbeResult JSON root must be an object")
        return cls.from_dict(cast(dict[str, object], parsed))


def _pop(data: dict[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"missing required field {key!r}")
    return data[key]


def _pop_str(data: dict[str, object], key: str) -> str:
    value = _pop(data, key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _pop_int(data: dict[str, object], key: str) -> int:
    return _as_int(_pop(data, key), key)


def _pop_float(data: dict[str, object], key: str) -> float:
    value = _pop(data, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _pop_bool(data: dict[str, object], key: str) -> bool:
    return _as_bool(_pop(data, key), key)


def _pop_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = _pop(data, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _pop_metadata(data: dict[str, object], key: str) -> Metadata:
    raw = _pop_dict(data, key)
    metadata: Metadata = {}
    for meta_key, value in raw.items():
        _validate_metadata_value(value, f"metadata[{meta_key!r}]")
        metadata[meta_key] = cast(MetadataValue, value)
    return metadata


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, name)


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _require_non_empty(value: str, name: str) -> None:
    if value == "":
        raise ValueError(f"{name} must be non-empty")


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_metadata_value(value: object, name: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    raise ValueError(f"{name} must be a scalar JSON value")
