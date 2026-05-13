"""Event-driven runner that exercises an Allocator over a workload.

The runner walks a min-heap of (tick, kind, request_id) tuples. Three
event kinds participate: ``departure`` (priority 0, processed first at a
given tick so that resources free before new requests land),
``growth`` (priority 1, fires when a decode step crosses a KV block
boundary), ``arrival`` (priority 2).

Time is virtual ticks; one tick maps to one decode step. The runner
never sleeps. Allocator calls are wall-clock timed via
``time.perf_counter_ns`` so latency stats reflect real allocator cost.
"""

from __future__ import annotations

import heapq
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

from cachepawl import __version__ as cachepawl_version
from cachepawl.allocator.base import Allocator
from cachepawl.benchmarks.harness.metrics import MetricsCollector
from cachepawl.benchmarks.harness.schema import (
    BenchmarkRun,
    Environment,
    Hardware,
)
from cachepawl.benchmarks.harness.workloads import (
    Request,
    WorkloadSpec,
    generate_request_stream,
)
from cachepawl.quant.dtypes import DType, bytes_per_element

_EVENT_DEPARTURE = 0
_EVENT_GROWTH = 1
_EVENT_ARRIVAL = 2

_DEFAULT_KV_BLOCK_TOKENS = 16
_DEFAULT_SAMPLE_EVERY_N_EVENTS = 100


@dataclass(frozen=True, slots=True, order=True)
class _Event:
    tick: int
    priority: int
    request_id: int


def run_benchmark(
    allocator: Allocator,
    spec: WorkloadSpec,
    *,
    allocator_name: str,
    output_dir: Path,
    device: str = "cpu",
    notes: str = "",
    kv_block_tokens: int = _DEFAULT_KV_BLOCK_TOKENS,
    sample_every_n_events: int = _DEFAULT_SAMPLE_EVERY_N_EVENTS,
    record_memory_snapshot: bool = False,
) -> BenchmarkRun:
    """Drive ``allocator`` through ``spec`` and write a BenchmarkRun JSON.

    Returns the in-memory BenchmarkRun; the same object is also serialized
    to ``output_dir/<allocator_name>/<workload_name>/<timestamp>.json``.
    """

    if kv_block_tokens <= 0:
        raise ValueError(f"kv_block_tokens must be positive, got {kv_block_tokens}")
    if sample_every_n_events <= 0:
        raise ValueError(f"sample_every_n_events must be positive, got {sample_every_n_events}")

    requests = generate_request_stream(spec)
    events, growth_ticks = _build_event_heap(requests, kv_block_tokens)
    dtype_bytes = _dtype_bytes_int(spec.dtype)

    started_at = _utc_now_iso()
    snapshot_path = _snapshot_path(output_dir, allocator_name, spec, started_at)

    active_blocks: dict[int, list[int]] = {}
    requests_by_id = {req.request_id: req for req in requests}
    event_count = 0

    with MetricsCollector(
        device=device,
        allocator=allocator,
        record_memory_snapshot=record_memory_snapshot,
        snapshot_path=snapshot_path,
    ) as collector:
        while events:
            event = heapq.heappop(events)
            event_count += 1
            request = requests_by_id[event.request_id]
            if event.priority == _EVENT_ARRIVAL:
                _process_arrival(
                    allocator=allocator,
                    collector=collector,
                    request=request,
                    spec=spec,
                    kv_block_tokens=kv_block_tokens,
                    dtype_bytes=dtype_bytes,
                    active_blocks=active_blocks,
                )
            elif event.priority == _EVENT_GROWTH:
                _process_growth(
                    allocator=allocator,
                    collector=collector,
                    request_id=request.request_id,
                    spec=spec,
                    dtype_bytes=dtype_bytes,
                    active_blocks=active_blocks,
                )
            else:
                _process_departure(
                    allocator=allocator,
                    collector=collector,
                    request_id=request.request_id,
                    active_blocks=active_blocks,
                )
            if event_count % sample_every_n_events == 0:
                collector.sample(num_active_requests=len(active_blocks))
        collector.sample(num_active_requests=len(active_blocks))

    finished_at = _utc_now_iso()
    run = BenchmarkRun(
        spec=spec,
        allocator_name=allocator_name,
        hardware=_capture_hardware(device),
        environment=_capture_environment(),
        started_at=started_at,
        finished_at=finished_at,
        metrics=collector.metrics,
        notes=_augment_notes(notes, growth_ticks=len(growth_ticks)),
    )
    target = _result_path(output_dir, allocator_name, spec, started_at)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run.to_json())
    return run


def _build_event_heap(
    requests: list[Request],
    kv_block_tokens: int,
) -> tuple[list[_Event], list[int]]:
    heap: list[_Event] = []
    growth_ticks: list[int] = []
    for req in requests:
        heapq.heappush(heap, _Event(req.arrival_tick, _EVENT_ARRIVAL, req.request_id))
        heapq.heappush(heap, _Event(req.departure_tick, _EVENT_DEPARTURE, req.request_id))
        for tick in _growth_ticks_for(req, kv_block_tokens):
            growth_ticks.append(tick)
            heapq.heappush(heap, _Event(tick, _EVENT_GROWTH, req.request_id))
    return heap, growth_ticks


def _growth_ticks_for(req: Request, kv_block_tokens: int) -> list[int]:
    """Ticks at which the KV cache crosses a block boundary mid-generation."""

    out: list[int] = []
    if req.prompt_len <= 0:
        return out
    total = req.prompt_len
    current_blocks = math.ceil(total / kv_block_tokens)
    for step in range(req.gen_len):
        total += 1
        needed = math.ceil(total / kv_block_tokens)
        if needed > current_blocks:
            current_blocks = needed
            out.append(req.arrival_tick + req.prompt_len + step + 1)
    return out


def _process_arrival(
    *,
    allocator: Allocator,
    collector: MetricsCollector,
    request: Request,
    spec: WorkloadSpec,
    kv_block_tokens: int,
    dtype_bytes: int,
    active_blocks: dict[int, list[int]],
) -> None:
    initial_kv_blocks_per_layer = max(1, math.ceil(request.prompt_len / kv_block_tokens))
    ids: list[int] = []
    for _layer_idx in range(spec.attention_layers):
        ids.extend(
            _timed_allocate(
                allocator,
                collector,
                initial_kv_blocks_per_layer,
                dtype_bytes,
            )
        )
    for _layer_idx in range(spec.ssm_layers):
        ids.extend(_timed_allocate(allocator, collector, 1, dtype_bytes))
    active_blocks[request.request_id] = ids


def _process_growth(
    *,
    allocator: Allocator,
    collector: MetricsCollector,
    request_id: int,
    spec: WorkloadSpec,
    dtype_bytes: int,
    active_blocks: dict[int, list[int]],
) -> None:
    held = active_blocks.get(request_id)
    if held is None:
        return
    for _layer_idx in range(spec.attention_layers):
        held.extend(_timed_allocate(allocator, collector, 1, dtype_bytes))


def _process_departure(
    *,
    allocator: Allocator,
    collector: MetricsCollector,
    request_id: int,
    active_blocks: dict[int, list[int]],
) -> None:
    held = active_blocks.pop(request_id, None)
    if held is None:
        return
    start = time.perf_counter_ns()
    try:
        allocator.free(held)
    except torch.cuda.OutOfMemoryError:
        collector.record_oom()
    elapsed = time.perf_counter_ns() - start
    collector.record_free(elapsed)


def _timed_allocate(
    allocator: Allocator,
    collector: MetricsCollector,
    num_blocks: int,
    dtype_bytes: int,
) -> list[int]:
    start = time.perf_counter_ns()
    try:
        ids = allocator.allocate(num_blocks, dtype_bytes=dtype_bytes)
    except torch.cuda.OutOfMemoryError:
        elapsed = time.perf_counter_ns() - start
        collector.record_allocate(elapsed)
        collector.record_oom()
        return []
    elapsed = time.perf_counter_ns() - start
    collector.record_allocate(elapsed)
    return ids


def _dtype_bytes_int(dtype: DType) -> int:
    """Round dtype width up to the nearest integer byte for the Allocator ABC."""

    return max(1, math.ceil(bytes_per_element(dtype)))


def _capture_hardware(device: str) -> Hardware:
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        return Hardware(
            device="cuda",
            gpu_name=str(gpu_name),
            vram_total_bytes=int(props.total_memory),
            cuda_capability=(int(props.major), int(props.minor)),
        )
    return Hardware(
        device=device,
        gpu_name=None,
        vram_total_bytes=None,
        cuda_capability=None,
    )


def _capture_environment() -> Environment:
    return Environment(
        torch_version=torch.__version__,
        numpy_version=numpy.__version__,
        cachepawl_version=cachepawl_version,
        cuda_version=torch.version.cuda,
        python_version=platform.python_version() or sys.version.split()[0],
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _result_path(
    output_dir: Path,
    allocator_name: str,
    spec: WorkloadSpec,
    started_at: str,
) -> Path:
    safe_stamp = started_at.replace(":", "-")
    return output_dir / allocator_name / spec.name / f"{safe_stamp}.json"


def _snapshot_path(
    output_dir: Path,
    allocator_name: str,
    spec: WorkloadSpec,
    started_at: str,
) -> str:
    safe_stamp = started_at.replace(":", "-")
    return str(output_dir / allocator_name / spec.name / f"{safe_stamp}-memory.pickle")


def _augment_notes(notes: str, *, growth_ticks: int) -> str:
    summary = f"growth_events={growth_ticks}"
    if notes:
        return f"{notes} [{summary}]"
    return summary
