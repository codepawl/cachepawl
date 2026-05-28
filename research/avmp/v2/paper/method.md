# Method

The Path C method is an observe/translate/replay/diagnose workflow. It is
designed to preserve the boundary between advisory analysis and runtime
mutation.

## 1. Observe Vanilla vLLM Artifacts

The workflow starts from a vanilla `vllm==0.21.0` run for
`Zyphra/Zamba2-2.7B-instruct`. Observation scripts collect cache-plan artifacts
and safe runtime metadata. The observation path is read-only: it does not edit
vLLM source, monkeypatch vLLM internals, replace allocators, or return
Cachepawl plans to vLLM.

The captured artifacts include the runtime cache plan, safe metadata about the
planner inputs, scheduler/KV manager structure, worker tensor layout, live
request/block assignment, and attention-side block-table/view metadata.

## 2. Translate Cache Plans

The runtime cache plan is translated into a Cachepawl schema that separates the
fields needed for advisory metrics from the fields that would be required for
future controlled substitution. The translation records cache group count,
cache tensor count, layer count, block count, per-group cache kind, page size,
block size, useful bytes, and observed reserved bytes.

This schema is the input to both the diagnostic CLI and the planner-stage
advisory comparison.

## 3. Replay the Planner Stage

The workflow replays `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` on real
planner inputs captured from the observed run. The replay is bounded and
post-initialization. It is used to check whether the planner-stage translation
matches the runtime scheduler cache configuration.

The current evidence records:

- `planner_matches_runtime_scheduler=true`;
- `runtime_changed_during_replay=false`.

This grounds the advisory comparison in the same cache configuration used by
the runtime scheduler, without changing that runtime scheduler state.

## 4. Compute Advisory Metrics

Cachepawl compares the observed reserved cache bytes with the useful byte
footprint implied by the translated hybrid cache plan. The diagnostic report
records:

- observed reserved bytes;
- observed useful bytes;
- Cachepawl recommended bytes for the advisory planner view;
- estimated advisory savings bytes;
- overestimation ratio;
- wasted fraction.

These are planner-level advisory metrics. They estimate over-reservation in the
observed planner artifacts. They are not serving-time VRAM measurements and do
not imply throughput, latency, or quality effects.

## 5. Package Artifact-Input Diagnosis

`cachepawl diagnose-vllm` is the productized output of the phase. It consumes an
existing translated cache config and optional raw safe metadata, then writes:

- `report.json`;
- `summary.md`;
- `manifest.json`.

The artifact-input CLI is intentionally lightweight. It does not require vLLM,
GPU access, CUDA, NVML, or model loading. It does not modify vLLM, replace
allocators, or enable runtime mutation.

## 6. Inspect Mutation Contracts

The final method step is not substitution. It is a contract audit that records
which runtime structures were safely observable and which were not.

Observed contracts:

- planner-stage replay matched runtime scheduler config;
- request-to-block assignment was observed;
- worker tensor layout was observed;
- attention block-table/view metadata was observed.

Blocked contracts:

- Mamba state-index contract;
- Mamba state tensor contract.

The current evidence therefore supports advisory diagnosis and blocks allocator
replacement claims.
