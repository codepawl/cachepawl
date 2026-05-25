"""Import-safe translators for vLLM cache planning objects.

The functions in this module accept duck-typed vLLM-like objects and never
import ``vllm``. They are intended for observe-first snapshots of vLLM cache
plans before any scheduler, manager, or allocator mutation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import prod
from typing import Literal, TypeAlias, cast

VllmObservedCacheKind: TypeAlias = Literal["attention", "mamba"]
JsonLike: TypeAlias = object

_MISSING = object()


class VllmTranslationError(TypeError):
    """Raised when a duck-typed vLLM cache object cannot be translated."""


@dataclass(frozen=True, slots=True)
class VllmTranslatedCacheSpec:
    """Cachepawl-owned snapshot of one vLLM cache spec."""

    cache_kind: VllmObservedCacheKind
    spec_type: str
    layer_name: str | None
    block_size: int | None
    page_size_bytes: int | None
    useful_bytes: int | None
    dtype: str | None
    metadata: dict[str, JsonLike] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "cache_kind": self.cache_kind,
            "spec_type": self.spec_type,
            "layer_name": self.layer_name,
            "block_size": self.block_size,
            "page_size_bytes": self.page_size_bytes,
            "useful_bytes": self.useful_bytes,
            "dtype": self.dtype,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class VllmTranslatedCacheGroup:
    """Cachepawl-owned snapshot of one vLLM cache group."""

    group_index: int
    group_name: str | None
    layer_names: tuple[str, ...]
    layer_count: int
    cache_spec: VllmTranslatedCacheSpec
    is_eagle_group: bool | None
    metadata: dict[str, JsonLike] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "group_index": self.group_index,
            "group_name": self.group_name,
            "layer_names": self.layer_names,
            "layer_count": self.layer_count,
            "cache_spec": self.cache_spec.to_dict(),
            "is_eagle_group": self.is_eagle_group,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class VllmTranslatedCacheTensor:
    """Cachepawl-owned snapshot of one planned vLLM cache tensor."""

    size_bytes: int
    shared_by: tuple[str, ...]
    metadata: dict[str, JsonLike] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "size_bytes": self.size_bytes,
            "shared_by": self.shared_by,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class VllmTranslatedCacheConfig:
    """Cachepawl-owned snapshot of a vLLM ``KVCacheConfig``."""

    num_blocks: int | None
    groups: tuple[VllmTranslatedCacheGroup, ...]
    tensors: tuple[VllmTranslatedCacheTensor, ...]
    block_size: int | None
    dtype: str | None
    metadata: dict[str, JsonLike] = field(default_factory=dict)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def layer_count(self) -> int:
        return sum(group.layer_count for group in self.groups)

    @property
    def attention_group_count(self) -> int:
        return sum(1 for group in self.groups if group.cache_spec.cache_kind == "attention")

    @property
    def mamba_group_count(self) -> int:
        return sum(1 for group in self.groups if group.cache_spec.cache_kind == "mamba")

    @property
    def total_page_size_bytes(self) -> int | None:
        return _sum_optional(group.cache_spec.page_size_bytes for group in self.groups)

    @property
    def total_useful_bytes(self) -> int | None:
        return _sum_optional(group.cache_spec.useful_bytes for group in self.groups)

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "group_count": self.group_count,
            "layer_count": self.layer_count,
            "attention_group_count": self.attention_group_count,
            "mamba_group_count": self.mamba_group_count,
            "total_page_size_bytes": self.total_page_size_bytes,
            "total_useful_bytes": self.total_useful_bytes,
            "groups": tuple(group.to_dict() for group in self.groups),
            "tensors": tuple(tensor.to_dict() for tensor in self.tensors),
            "metadata": dict(sorted(self.metadata.items())),
        }


def translate_kv_cache_spec(layer_name: str | None, spec: object) -> VllmTranslatedCacheSpec:
    """Translate a vLLM-like ``KVCacheSpec`` object into a typed snapshot."""

    cache_kind = _cache_kind(spec)
    if cache_kind is None:
        raise VllmTranslationError(
            f"unsupported vLLM cache spec {type(spec).__name__!r}; "
            "expected an attention or mamba/state cache spec"
        )

    block_size = _optional_int(_get(spec, "block_size"), "block_size")
    page_size_bytes = _optional_int(_get(spec, "page_size_bytes"), "page_size_bytes")
    dtype = _spec_dtype(spec)
    useful_bytes = _useful_bytes(
        spec,
        cache_kind=cache_kind,
        block_size=block_size,
        page_size_bytes=page_size_bytes,
        dtype=dtype,
    )
    metadata = _spec_metadata(spec, cache_kind=cache_kind)

    return VllmTranslatedCacheSpec(
        cache_kind=cache_kind,
        spec_type=type(spec).__name__,
        layer_name=layer_name,
        block_size=block_size,
        page_size_bytes=page_size_bytes,
        useful_bytes=useful_bytes,
        dtype=dtype,
        metadata=metadata,
    )


def translate_kv_cache_group(group_index: int, group: object) -> VllmTranslatedCacheGroup:
    """Translate a vLLM-like ``KVCacheGroupSpec`` object."""

    if group_index < 0:
        raise ValueError("group_index must be non-negative")

    layer_names = _string_tuple(_get(group, "layer_names", "layers", "layer_names_in_group"))
    spec = _get(group, "kv_cache_spec", "cache_spec", "spec")
    if spec is _MISSING:
        raise VllmTranslationError(
            f"unsupported vLLM cache group {type(group).__name__!r}; missing cache spec"
        )
    group_name = _optional_string(_get(group, "group_id", "group_name", "layer_group_name"))
    translated_spec = translate_kv_cache_spec(group_name, spec)
    is_eagle_group = _optional_bool(_get(group, "is_eagle_group"), "is_eagle_group")
    metadata = _metadata_from_attrs(group, ("group_id", "group_name", "layer_group_name"))

    return VllmTranslatedCacheGroup(
        group_index=group_index,
        group_name=group_name,
        layer_names=layer_names,
        layer_count=len(layer_names),
        cache_spec=translated_spec,
        is_eagle_group=is_eagle_group,
        metadata=metadata,
    )


def translate_kv_cache_tensor(tensor: object) -> VllmTranslatedCacheTensor:
    """Translate a vLLM-like ``KVCacheTensor`` object."""

    size = _optional_int(_get(tensor, "size", "size_bytes"), "size")
    if size is None:
        raise VllmTranslationError(
            f"unsupported vLLM cache tensor {type(tensor).__name__!r}; missing size"
        )
    shared_by = _string_tuple(_get(tensor, "shared_by", "shared_layer_names", "layer_names"))
    return VllmTranslatedCacheTensor(size_bytes=size, shared_by=shared_by)


def translate_kv_cache_config(config: object) -> VllmTranslatedCacheConfig:
    """Translate a vLLM-like ``KVCacheConfig`` object."""

    raw_groups = _get(config, "kv_cache_groups", "cache_groups", "groups")
    if raw_groups is _MISSING:
        raise VllmTranslationError(
            f"unsupported vLLM cache config {type(config).__name__!r}; missing cache groups"
        )
    groups = tuple(
        translate_kv_cache_group(index, group)
        for index, group in enumerate(_iterable(raw_groups, "kv_cache_groups"))
    )

    raw_tensors = _get(config, "kv_cache_tensors", "cache_tensors", "tensors")
    tensors = (
        tuple(
            translate_kv_cache_tensor(tensor)
            for tensor in _iterable(raw_tensors, "kv_cache_tensors")
        )
        if raw_tensors is not _MISSING
        else ()
    )

    metadata = _metadata_from_attrs(
        config,
        (
            "hash_block_size",
            "mamba_block_size",
            "mamba_cache_mode",
            "mamba_page_size_padded",
            "gpu_memory_utilization",
            "kv_cache_memory_bytes",
        ),
    )

    return VllmTranslatedCacheConfig(
        num_blocks=_optional_int(_get(config, "num_blocks"), "num_blocks"),
        groups=groups,
        tensors=tensors,
        block_size=_optional_int(_get(config, "block_size"), "block_size"),
        dtype=_optional_string(_get(config, "cache_dtype", "dtype")),
        metadata=metadata,
    )


def _cache_kind(spec: object) -> VllmObservedCacheKind | None:
    type_name = type(spec).__name__.lower()
    if "attention" in type_name or _get(spec, "num_kv_heads", "head_size") is not _MISSING:
        return "attention"
    has_mamba_attrs = _get(spec, "shapes", "mamba_type", "mamba_cache_mode") is not _MISSING
    if "mamba" in type_name or has_mamba_attrs:
        return "mamba"
    return None


def _spec_dtype(spec: object) -> str | None:
    dtype = _optional_string(_get(spec, "dtype"))
    if dtype is not None:
        return dtype
    dtypes = _get(spec, "dtypes")
    if isinstance(dtypes, dict):
        unique = {str(value) for value in dtypes.values()}
        if len(unique) == 1:
            return next(iter(unique))
    return None


def _useful_bytes(
    spec: object,
    *,
    cache_kind: VllmObservedCacheKind,
    block_size: int | None,
    page_size_bytes: int | None,
    dtype: str | None,
) -> int | None:
    real_page = _optional_int(
        _get(spec, "real_page_size_bytes", "logical_page_size_bytes", "useful_bytes"),
        "real_page_size_bytes",
    )
    if real_page is not None:
        return real_page
    if cache_kind == "attention":
        return _derive_attention_bytes(spec, block_size=block_size, fallback_dtype=dtype)
    return _derive_mamba_bytes(spec) or page_size_bytes


def _derive_attention_bytes(
    spec: object, *, block_size: int | None, fallback_dtype: str | None
) -> int | None:
    num_kv_heads = _optional_int(_get(spec, "num_kv_heads"), "num_kv_heads")
    head_size = _optional_int(_get(spec, "head_size", "head_dim"), "head_size")
    dtype = _optional_string(_get(spec, "dtype")) or fallback_dtype
    if None in (num_kv_heads, head_size, block_size, dtype):
        return None
    dtype_bytes = _dtype_size_bytes(cast(str, dtype))
    if dtype_bytes is None:
        return None
    return 2 * cast(int, num_kv_heads) * cast(int, head_size) * cast(int, block_size) * dtype_bytes


def _derive_mamba_bytes(spec: object) -> int | None:
    shapes = _get(spec, "shapes")
    dtypes = _get(spec, "dtypes")
    if isinstance(shapes, dict):
        return _derive_mamba_bytes_from_mapping(shapes, dtypes)
    if isinstance(shapes, tuple | list):
        return _derive_mamba_bytes_from_sequence(shapes, dtypes)
    return None


def _derive_mamba_bytes_from_mapping(shapes: dict[object, object], dtypes: object) -> int | None:
    total = 0
    for name, raw_shape in shapes.items():
        shape = _shape_tuple(raw_shape)
        if shape is None:
            return None
        dtype_name = _dtype_for_shape(name, dtypes)
        dtype_bytes = _dtype_size_bytes(dtype_name) if dtype_name is not None else None
        if dtype_bytes is None:
            return None
        total += prod(shape) * dtype_bytes
    return total


def _derive_mamba_bytes_from_sequence(
    shapes: tuple[object, ...] | list[object], dtypes: object
) -> int | None:
    if not isinstance(dtypes, tuple | list) or len(shapes) != len(dtypes):
        return None
    total = 0
    for raw_shape, dtype in zip(shapes, dtypes, strict=True):
        shape = _shape_tuple(raw_shape)
        if shape is None:
            return None
        dtype_bytes = _dtype_size_bytes(str(dtype))
        if dtype_bytes is None:
            return None
        total += prod(shape) * dtype_bytes
    return total


def _dtype_for_shape(name: object, dtypes: object) -> str | None:
    if isinstance(dtypes, dict):
        if name in dtypes:
            return str(dtypes[name])
        if len(dtypes) == 1:
            return str(next(iter(dtypes.values())))
    return None


def _dtype_size_bytes(dtype: str) -> int | None:
    normalized = dtype.lower().replace("torch.", "")
    if normalized in {"float16", "bfloat16", "half", "fp16", "bf16"}:
        return 2
    if normalized in {"float32", "fp32", "float", "single"}:
        return 4
    if normalized in {"float64", "fp64", "double"}:
        return 8
    if normalized in {"int8", "uint8", "bool"}:
        return 1
    if normalized in {"int16", "uint16"}:
        return 2
    if normalized in {"int32", "uint32"}:
        return 4
    if normalized in {"int64", "uint64"}:
        return 8
    return None


def _spec_metadata(spec: object, *, cache_kind: VllmObservedCacheKind) -> dict[str, JsonLike]:
    names: tuple[str, ...]
    if cache_kind == "attention":
        names = (
            "num_kv_heads",
            "head_size",
            "head_dim",
            "kv_quant_mode",
            "page_size_padded",
            "storage_block_size",
            "max_memory_usage_bytes",
        )
    else:
        names = (
            "shapes",
            "dtypes",
            "mamba_type",
            "mamba_cache_mode",
            "num_speculative_blocks",
            "page_size_padded",
            "storage_block_size",
            "max_memory_usage_bytes",
        )
    return _metadata_from_attrs(spec, names)


def _metadata_from_attrs(obj: object, names: tuple[str, ...]) -> dict[str, JsonLike]:
    metadata: dict[str, JsonLike] = {}
    for name in names:
        value = _get(obj, name)
        if value is not _MISSING and value is not None and not callable(value):
            metadata[name] = _to_json_like(value)
    return metadata


def _get(obj: object, *names: str) -> object:
    for name in names:
        try:
            return getattr(obj, name)
        except AttributeError:
            continue
    return _MISSING


def _optional_int(value: object, name: str) -> int | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise VllmTranslationError(f"{name} must be an integer when provided")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, bool):
        raise VllmTranslationError(f"{name} must be a boolean when provided")
    return value


def _optional_string(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    return str(value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is _MISSING or value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in _iterable(value, "layer names"))


def _iterable(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, dict):
        return tuple(value.values())
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise VllmTranslationError(f"{name} must be iterable") from exc


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    if isinstance(value, int):
        return (value,)
    try:
        shape: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in shape):
        return None
    return cast(tuple[int, ...], shape)


def _to_json_like(value: object) -> JsonLike:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {
            str(key): _to_json_like(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return tuple(_to_json_like(item) for item in value)
    return str(value)


def _sum_optional(values: Iterable[int | None]) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total
