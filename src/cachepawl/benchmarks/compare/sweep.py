"""Multi-cell sweep runner.

Walks the cartesian product of allocator variant x workload x model spec x
pool size x seed replicate and dispatches each cell to ``run_benchmark``.
Per-cell exceptions are captured and recorded; one bad cell does not abort
the sweep.

Per-cell JSON files land at::

    <output_dir>/runs/<variant_label>/<workload_name>/<stem>.json

where stem is ``<model_spec>__tb<size_human>__seed<seed>``. The internal
runner also writes a timestamp-named JSON via its own side effect; we
redirect that to a temporary scratch directory that is discarded at sweep
end. The canonical files in ``runs/`` are written by this module from the
in-memory ``BenchmarkRun`` returned by ``run_benchmark``.

This module bypasses ``cachepawl.benchmarks.REGISTRY`` because that
registry's factory signature bakes a fixed ``total_bytes`` and a
workload-derived model spec. The sweep needs to vary both axes, so it
constructs ``PaddedUnifiedPool`` and ``FixedDualPool`` directly. The
variant labels still match the registry's names plus a kwargs suffix.
"""

from __future__ import annotations

import dataclasses
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

from cachepawl import __version__ as cachepawl_version
from cachepawl.allocator.base import Allocator
from cachepawl.allocator.baselines import FixedDualPool, PaddedUnifiedPool
from cachepawl.benchmarks import PRESETS, BenchmarkRun, run_benchmark
from cachepawl.models.spec import JAMBA_1_5_MINI_REF, MAMBA2_1B3_REF, HybridModelSpec

# Stem grammar: <model_spec_slug>__tb<size_human>__seed<seed>
# Two underscores between fields, single underscores within model spec
# slug. Update _cell_stem and _parse_cell_stem in lockstep.
CELL_STEM_PATTERN: str = (
    r"^(?P<model_spec>[a-z0-9_]+?)__tb(?P<tb>[0-9a-z]+)__seed(?P<seed>-?[0-9]+)$"
)

# CPU-only fallback when total_bytes is too small to fit even one page.
# All cells use this as their hard floor.
_MIN_TOTAL_BYTES: int = 64 * 1024 * 1024  # 64 MiB


@dataclass(frozen=True, slots=True)
class AllocatorVariant:
    """A specific allocator plus the constructor kwargs to sweep with.

    ``label`` is the identifier that appears in filenames, the report,
    and plot legends. ``allocator_name`` selects the underlying class
    (one of ``padded_unified`` or ``fixed_dual``). ``kwargs`` is a tuple
    of (name, value) pairs so the dataclass stays hashable.
    """

    label: str
    allocator_name: str
    kwargs: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """All inputs to one sweep invocation.

    ``smoke_num_requests`` is an internal override that shrinks every
    workload's request count for the ``--smoke`` test path. Production
    sweeps leave it ``None``.
    """

    variants: tuple[AllocatorVariant, ...]
    workload_names: tuple[str, ...]
    model_spec_names: tuple[str, ...]
    total_bytes_options: tuple[int, ...]
    device: str
    output_dir: Path
    seed_replicates: int = 3
    smoke_num_requests: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "variants": [_variant_to_dict(v) for v in self.variants],
            "workload_names": list(self.workload_names),
            "model_spec_names": list(self.model_spec_names),
            "total_bytes_options": list(self.total_bytes_options),
            "device": self.device,
            "output_dir": str(self.output_dir),
            "seed_replicates": self.seed_replicates,
            "smoke_num_requests": self.smoke_num_requests,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SweepConfig:
        variants = tuple(_variant_from_dict(_as_dict(item)) for item in _as_list(data["variants"]))
        return cls(
            variants=variants,
            workload_names=tuple(_as_str(x) for x in _as_list(data["workload_names"])),
            model_spec_names=tuple(_as_str(x) for x in _as_list(data["model_spec_names"])),
            total_bytes_options=tuple(_as_int(x) for x in _as_list(data["total_bytes_options"])),
            device=_as_str(data["device"]),
            output_dir=Path(_as_str(data["output_dir"])),
            seed_replicates=_as_int(data["seed_replicates"]),
            smoke_num_requests=_as_optional_int(data.get("smoke_num_requests")),
        )


@dataclass(slots=True)
class CellFailure:
    """A single cell that raised during execution."""

    variant_label: str
    workload_name: str
    model_spec_name: str
    total_bytes: int
    seed: int
    exception_repr: str
    elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_label": self.variant_label,
            "workload_name": self.workload_name,
            "model_spec_name": self.model_spec_name,
            "total_bytes": self.total_bytes,
            "seed": self.seed,
            "exception_repr": self.exception_repr,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True, slots=True)
class SweepMetadata:
    """Provenance captured around a single sweep invocation.

    All version strings and the git SHA are captured once at sweep start
    and held constant for all cells. ``hardware_label`` is a short human
    string (for example ``"cpu (linux x86_64)"``) used in the report
    footer and plot watermark; the structured fields below are the
    source of truth.
    """

    git_sha: str
    torch_version: str
    numpy_version: str
    python_version: str
    cachepawl_version: str
    cuda_version: str | None
    gpu_name: str | None
    device: str
    hardware_label: str
    sweep_started_at: str
    sweep_finished_at: str
    total_wall_seconds: float
    n_cells_planned: int
    n_cells_succeeded: int
    n_cells_failed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "python_version": self.python_version,
            "cachepawl_version": self.cachepawl_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "device": self.device,
            "hardware_label": self.hardware_label,
            "sweep_started_at": self.sweep_started_at,
            "sweep_finished_at": self.sweep_finished_at,
            "total_wall_seconds": self.total_wall_seconds,
            "n_cells_planned": self.n_cells_planned,
            "n_cells_succeeded": self.n_cells_succeeded,
            "n_cells_failed": self.n_cells_failed,
        }


@dataclass(slots=True)
class SweepResult:
    """In-memory result of one sweep invocation.

    ``runs`` is in order of cell execution. ``cell_stems`` maps the
    index in ``runs`` to the canonical filename stem so callers can
    locate each on-disk artifact without re-deriving the stem.
    """

    config: SweepConfig
    runs: list[BenchmarkRun]
    failures: list[CellFailure]
    metadata: SweepMetadata
    cell_stems: dict[int, str] = field(default_factory=dict)


DEFAULT_VARIANTS: tuple[AllocatorVariant, ...] = (
    AllocatorVariant(label="padded_unified", allocator_name="padded_unified", kwargs=()),
    AllocatorVariant(
        label="fixed_dual_mr05",
        allocator_name="fixed_dual",
        kwargs=(("mamba_ratio", 0.5),),
    ),
    AllocatorVariant(
        label="fixed_dual_mr09",
        allocator_name="fixed_dual",
        kwargs=(("mamba_ratio", 0.9),),
    ),
)

DEFAULT_WORKLOAD_NAMES: tuple[str, ...] = ("uniform_short", "mixed_long", "agentic_burst")
DEFAULT_MODEL_SPEC_NAMES: tuple[str, ...] = ("jamba_1_5_mini", "mamba2_1b3")
DEFAULT_TOTAL_BYTES_OPTIONS: tuple[int, ...] = (
    1 * 1024**3,
    4 * 1024**3,
    8 * 1024**3,
)
DEFAULT_SEED_REPLICATES: int = 3

QUICK_WORKLOAD_NAMES: tuple[str, ...] = ("uniform_short",)
QUICK_MODEL_SPEC_NAMES: tuple[str, ...] = ("jamba_1_5_mini",)
QUICK_TOTAL_BYTES_OPTIONS: tuple[int, ...] = (1 * 1024**3,)
QUICK_SEED_REPLICATES: int = 1

SMOKE_NUM_REQUESTS: int = 16

_MODEL_SPECS: Mapping[str, HybridModelSpec] = {
    "jamba_1_5_mini": JAMBA_1_5_MINI_REF,
    "mamba2_1b3": MAMBA2_1B3_REF,
}


def known_model_spec_names() -> tuple[str, ...]:
    return tuple(_MODEL_SPECS.keys())


def get_model_spec(name: str) -> HybridModelSpec:
    if name not in _MODEL_SPECS:
        raise KeyError(f"unknown model_spec {name!r}; known: {sorted(_MODEL_SPECS)}")
    return _MODEL_SPECS[name]


def make_default_config(output_dir: Path, device: str) -> SweepConfig:
    """Full sweep: 3 variants x 3 workloads x 2 specs x 3 sizes x 3 seeds = 162 cells."""

    return SweepConfig(
        variants=DEFAULT_VARIANTS,
        workload_names=DEFAULT_WORKLOAD_NAMES,
        model_spec_names=DEFAULT_MODEL_SPEC_NAMES,
        total_bytes_options=DEFAULT_TOTAL_BYTES_OPTIONS,
        device=device,
        output_dir=output_dir,
        seed_replicates=DEFAULT_SEED_REPLICATES,
        smoke_num_requests=None,
    )


def make_quick_config(output_dir: Path, device: str) -> SweepConfig:
    """Quick sweep used in CI and committed reference data (3 cells)."""

    return SweepConfig(
        variants=DEFAULT_VARIANTS,
        workload_names=QUICK_WORKLOAD_NAMES,
        model_spec_names=QUICK_MODEL_SPEC_NAMES,
        total_bytes_options=QUICK_TOTAL_BYTES_OPTIONS,
        device=device,
        output_dir=output_dir,
        seed_replicates=QUICK_SEED_REPLICATES,
        smoke_num_requests=None,
    )


def make_smoke_config(output_dir: Path, device: str) -> SweepConfig:
    """Single-cell sweep with a tiny workload, for the CLI smoke test."""

    return SweepConfig(
        variants=(DEFAULT_VARIANTS[0],),
        workload_names=QUICK_WORKLOAD_NAMES,
        model_spec_names=QUICK_MODEL_SPEC_NAMES,
        total_bytes_options=QUICK_TOTAL_BYTES_OPTIONS,
        device=device,
        output_dir=output_dir,
        seed_replicates=1,
        smoke_num_requests=SMOKE_NUM_REQUESTS,
    )


@dataclass(frozen=True, slots=True)
class _Cell:
    variant: AllocatorVariant
    workload_name: str
    model_spec_name: str
    total_bytes: int
    seed: int


def run_sweep(config: SweepConfig) -> SweepResult:
    """Execute every cell in ``config`` and return the aggregated result.

    The caller is responsible for persisting any further artifacts
    (aggregated JSON, report markdown, plots). ``run_sweep`` only writes
    the per-cell JSONs and the in-memory ``SweepResult``.
    """

    _validate_config(config)

    runs: list[BenchmarkRun] = []
    failures: list[CellFailure] = []
    cell_stems: dict[int, str] = {}

    cells = _enumerate_cells(config)
    total_cells = len(cells)

    output_runs_root = config.output_dir / "runs"
    output_runs_root.mkdir(parents=True, exist_ok=True)

    sweep_started_iso = _utc_now_iso()
    sweep_perf_start = time.perf_counter()

    device = torch.device(config.device)
    with tempfile.TemporaryDirectory(prefix="cp_compare_scratch_") as scratch:
        scratch_dir = Path(scratch)
        for idx, cell in enumerate(cells, start=1):
            cell_start = time.perf_counter()
            try:
                run = _execute_cell(cell, scratch_dir, config, device)
            except Exception as exc:
                elapsed = time.perf_counter() - cell_start
                failures.append(
                    CellFailure(
                        variant_label=cell.variant.label,
                        workload_name=cell.workload_name,
                        model_spec_name=cell.model_spec_name,
                        total_bytes=cell.total_bytes,
                        seed=cell.seed,
                        exception_repr=repr(exc)[:200],
                        elapsed_s=elapsed,
                    )
                )
                _print_progress_fail(idx, total_cells, cell, elapsed, exc)
                continue
            elapsed = time.perf_counter() - cell_start
            stem = _cell_stem(cell)
            canonical = output_runs_root / cell.variant.label / cell.workload_name / f"{stem}.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(run.to_json())
            cell_stems[len(runs)] = stem
            runs.append(run)
            _print_progress_ok(idx, total_cells, cell, elapsed)

    sweep_finished_iso = _utc_now_iso()
    total_wall = time.perf_counter() - sweep_perf_start

    metadata = _capture_metadata(
        config=config,
        sweep_started_iso=sweep_started_iso,
        sweep_finished_iso=sweep_finished_iso,
        total_wall_seconds=total_wall,
        n_planned=total_cells,
        n_succeeded=len(runs),
        n_failed=len(failures),
    )

    return SweepResult(
        config=config,
        runs=runs,
        failures=failures,
        metadata=metadata,
        cell_stems=cell_stems,
    )


def _execute_cell(
    cell: _Cell,
    scratch_dir: Path,
    config: SweepConfig,
    device: torch.device,
) -> BenchmarkRun:
    preset = PRESETS[cell.workload_name]
    model_spec = _MODEL_SPECS[cell.model_spec_name]
    cell_spec = dataclasses.replace(
        preset,
        seed=cell.seed,
        attention_profile=model_spec.attention_profile,
        ssm_profile=model_spec.ssm_profile,
    )
    if config.smoke_num_requests is not None:
        cell_spec = dataclasses.replace(cell_spec, num_requests=config.smoke_num_requests)
    allocator = _build_allocator(cell.variant, model_spec, cell.total_bytes, device)
    return run_benchmark(
        allocator=allocator,
        spec=cell_spec,
        allocator_name=cell.variant.label,
        output_dir=scratch_dir,
        device=config.device,
    )


def _build_allocator(
    variant: AllocatorVariant,
    model_spec: HybridModelSpec,
    total_bytes: int,
    device: torch.device,
) -> Allocator:
    kwargs = dict(variant.kwargs)
    if variant.allocator_name == "padded_unified":
        if kwargs:
            raise ValueError(f"padded_unified accepts no extra kwargs; got {sorted(kwargs)}")
        return PaddedUnifiedPool(
            model_spec=model_spec,
            total_bytes=total_bytes,
            device=device,
        )
    if variant.allocator_name == "fixed_dual":
        mamba_ratio = float(kwargs.pop("mamba_ratio", 0.5))
        if kwargs:
            raise ValueError(
                f"fixed_dual: unsupported kwargs {sorted(kwargs)}; only 'mamba_ratio' is recognized"
            )
        return FixedDualPool(
            model_spec=model_spec,
            total_bytes=total_bytes,
            device=device,
            mamba_ratio=mamba_ratio,
        )
    raise ValueError(
        f"unknown allocator_name {variant.allocator_name!r}; "
        "supported: 'padded_unified', 'fixed_dual'"
    )


def _enumerate_cells(config: SweepConfig) -> list[_Cell]:
    cells: list[_Cell] = []
    for variant in config.variants:
        for workload_name in config.workload_names:
            base_seed = PRESETS[workload_name].seed
            for model_spec_name in config.model_spec_names:
                for total_bytes in config.total_bytes_options:
                    for replicate_idx in range(config.seed_replicates):
                        cells.append(
                            _Cell(
                                variant=variant,
                                workload_name=workload_name,
                                model_spec_name=model_spec_name,
                                total_bytes=total_bytes,
                                seed=base_seed + replicate_idx,
                            )
                        )
    return cells


def _validate_config(config: SweepConfig) -> None:
    if config.device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {config.device!r}")
    if config.seed_replicates <= 0:
        raise ValueError(f"seed_replicates must be positive, got {config.seed_replicates}")
    if not config.variants:
        raise ValueError("variants must not be empty")
    if not config.workload_names:
        raise ValueError("workload_names must not be empty")
    if not config.model_spec_names:
        raise ValueError("model_spec_names must not be empty")
    if not config.total_bytes_options:
        raise ValueError("total_bytes_options must not be empty")
    for workload_name in config.workload_names:
        if workload_name not in PRESETS:
            raise ValueError(f"unknown workload_name {workload_name!r}; known: {sorted(PRESETS)}")
    for model_spec_name in config.model_spec_names:
        if model_spec_name not in _MODEL_SPECS:
            raise ValueError(
                f"unknown model_spec_name {model_spec_name!r}; known: {sorted(_MODEL_SPECS)}"
            )
    for total_bytes in config.total_bytes_options:
        if total_bytes < _MIN_TOTAL_BYTES:
            raise ValueError(f"total_bytes {total_bytes} below the {_MIN_TOTAL_BYTES}-byte floor")
    for variant in config.variants:
        if variant.allocator_name not in {"padded_unified", "fixed_dual"}:
            raise ValueError(
                f"variant {variant.label!r}: unknown allocator_name {variant.allocator_name!r}"
            )


def _cell_stem(cell: _Cell) -> str:
    return f"{cell.model_spec_name}__tb{_total_bytes_human(cell.total_bytes)}__seed{cell.seed}"


def total_bytes_human(value: int) -> str:
    """Render ``value`` as the short label used in filenames and tables."""

    return _total_bytes_human(value)


def _total_bytes_human(value: int) -> str:
    gib = 1024**3
    mib = 1024**2
    if value >= gib and value % gib == 0:
        return f"{value // gib}gib"
    if value >= mib and value % mib == 0:
        return f"{value // mib}mib"
    return f"{value}b"


def _capture_metadata(
    *,
    config: SweepConfig,
    sweep_started_iso: str,
    sweep_finished_iso: str,
    total_wall_seconds: float,
    n_planned: int,
    n_succeeded: int,
    n_failed: int,
) -> SweepMetadata:
    gpu_name: str | None = None
    if config.device == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        except RuntimeError:
            gpu_name = None
    hardware_label = f"{config.device} ({platform.system().lower()} {platform.machine()})"
    return SweepMetadata(
        git_sha=_resolve_git_sha(),
        torch_version=str(torch.__version__),
        numpy_version=str(numpy.__version__),
        python_version=platform.python_version(),
        cachepawl_version=cachepawl_version,
        cuda_version=torch.version.cuda,
        gpu_name=gpu_name,
        device=config.device,
        hardware_label=hardware_label,
        sweep_started_at=sweep_started_iso,
        sweep_finished_at=sweep_finished_iso,
        total_wall_seconds=total_wall_seconds,
        n_cells_planned=n_planned,
        n_cells_succeeded=n_succeeded,
        n_cells_failed=n_failed,
    )


def _resolve_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _print_progress_ok(idx: int, total: int, cell: _Cell, elapsed_s: float) -> None:
    line = _progress_prefix(idx, total, cell, elapsed_s) + " | OK"
    print(line, file=sys.stdout, flush=True)


def _print_progress_fail(
    idx: int,
    total: int,
    cell: _Cell,
    elapsed_s: float,
    exc: BaseException,
) -> None:
    line = _progress_prefix(idx, total, cell, elapsed_s) + f" | FAILED: {repr(exc)[:60]}"
    print(line, file=sys.stderr, flush=True)


def _progress_prefix(idx: int, total: int, cell: _Cell, elapsed_s: float) -> str:
    tb_label = _total_bytes_human(cell.total_bytes)
    width = max(3, len(str(total)))
    return (
        f"[{idx:0{width}d}/{total:0{width}d}] {cell.variant.label} | "
        f"{cell.workload_name} | {cell.model_spec_name} | {tb_label} | "
        f"seed={cell.seed} | {elapsed_s:.2f}s"
    )


def _variant_to_dict(variant: AllocatorVariant) -> dict[str, object]:
    return {
        "label": variant.label,
        "allocator_name": variant.allocator_name,
        "kwargs": [[name, value] for name, value in variant.kwargs],
    }


def _variant_from_dict(data: Mapping[str, object]) -> AllocatorVariant:
    raw_kwargs = _as_list(data["kwargs"])
    kwargs: list[tuple[str, float]] = []
    for raw_pair in raw_kwargs:
        pair = _as_list(raw_pair)
        if len(pair) != 2:
            raise ValueError(f"kwargs pair must have length 2, got {pair!r}")
        kwargs.append((_as_str(pair[0]), _as_float(pair[1])))
    return AllocatorVariant(
        label=_as_str(data["label"]),
        allocator_name=_as_str(data["allocator_name"]),
        kwargs=tuple(kwargs),
    )


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("expected float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected float, got {type(value).__name__}")


def _as_list(value: object) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value).__name__}")
    return value


def _as_dict(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected dict, got {type(value).__name__}")
    return value


__all__ = [
    "CELL_STEM_PATTERN",
    "DEFAULT_MODEL_SPEC_NAMES",
    "DEFAULT_SEED_REPLICATES",
    "DEFAULT_TOTAL_BYTES_OPTIONS",
    "DEFAULT_VARIANTS",
    "DEFAULT_WORKLOAD_NAMES",
    "QUICK_MODEL_SPEC_NAMES",
    "QUICK_SEED_REPLICATES",
    "QUICK_TOTAL_BYTES_OPTIONS",
    "QUICK_WORKLOAD_NAMES",
    "SMOKE_NUM_REQUESTS",
    "AllocatorVariant",
    "CellFailure",
    "SweepConfig",
    "SweepMetadata",
    "SweepResult",
    "get_model_spec",
    "known_model_spec_names",
    "make_default_config",
    "make_quick_config",
    "make_smoke_config",
    "run_sweep",
    "total_bytes_human",
]
