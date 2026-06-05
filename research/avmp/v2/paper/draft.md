# Cachepawl Path C: Planner-Level Advisory Diagnosis of Hybrid Attention/Mamba Cache Overestimation

## Abstract

Hybrid Attention/Mamba language models stress inference cache planners because
attention KV pages and Mamba state caches have different shapes and reuse
contracts. Cachepawl Path C evaluates whether these planner-level inefficiencies
can be diagnosed from observed vLLM cache-plan artifacts without modifying the
serving stack.

We present an observe/advisory workflow that captures vanilla `vllm==0.21.0`
cache artifacts for `Zyphra/Zamba2-2.7B-instruct`, translates them into a
Cachepawl schema, replays the vLLM planner stage on real planner inputs, and
emits an advisory report through `cachepawl diagnose-vllm`. The planner-stage
replay matched the runtime scheduler cache configuration and did not change the
runtime scheduler state. Across a bounded 4-cell matrix for one model
(`max_model_len` in `{2048, 4096}`, `gpu_memory_utilization` in `{0.6, 0.7}`,
and `max_num_seqs=1`), estimated advisory savings ranged from `685,011,456` to
`1,347,563,520` bytes. The overestimation ratio stayed
`1.7333734577189286` and the wasted fraction stayed `0.4230902777777778`
across completed cells.

We also include a separate RTX 3060 planner-only pack for synthetic hybrid
KV/State workloads. In that deterministic pack, `cachepawl-avmp` stays closer
to useful bytes than the `vllm-style-padded` planner baseline, but the result is
not a vLLM serving measurement.

The artifact also records runtime contracts needed before any mutation attempt.
Live request-to-block assignment, worker tensor layout, and attention
block-table/view metadata were observed. Mamba state-index and Mamba state
tensor contracts remain blocked: `mamba_state_idx` was reachable but empty for
the live request, no Mamba state tensors were safely reachable, and the observed
cache config used `mamba_cache_mode: none`.

This is a diagnostic/advisory systems artifact. It does not claim runtime cache
substitution, allocator replacement, serving-time VRAM reduction, throughput
improvement, latency improvement, accuracy improvement, or quality impact. Its
contribution is a non-invasive method, evidence package, and product surface
for reporting planner-level hybrid cache overestimation while preserving
explicit gates for future controlled substitution.

## 1. Introduction

Hybrid Attention/Mamba language models place two different cache shapes behind
one inference runtime. Attention layers use KV blocks whose economics are tied
to token sequence growth, while Mamba or SSM layers use state tensors whose
useful byte footprint can diverge from the page shape used by a uniform cache
planner. A cache planner that reserves by a shared page/block abstraction can
therefore allocate capacity according to a shape that is larger than the useful
state payload for hybrid layouts.

Cachepawl Path C studies this gap as an observe/advisory problem. The goal of
this phase is not to replace vLLM's allocator, rewrite live runtime state, or
report serving results. Instead, the goal is to answer a narrower systems
question: can Cachepawl observe a vanilla vLLM hybrid cache plan, translate it
into a stable schema, replay the planner stage on the observed inputs, and emit
a practical diagnostic report that quantifies planner-level overestimation?

### 1.1 Scope of This Version

This version is scoped to evidence-backed diagnosis. It includes a bounded
Path C matrix for one observed `Zyphra/Zamba2-2.7B-instruct` setup, a separate
deterministic RTX 3060 planner-only comparison for synthetic `jamba-1.5-mini`
workloads, and the `cachepawl diagnose-vllm` artifact-input CLI. It excludes
runtime allocator replacement, runtime cache substitution, live VRAM reduction,
serving throughput, serving latency, quality, accuracy, and controlled
substitution readiness.

The answer from the current artifact set is yes, within a bounded scope. The
evaluation uses `Zyphra/Zamba2-2.7B-instruct` under vanilla `vllm==0.21.0` and a
4-cell advisory matrix with `max_model_len` in `{2048, 4096}`,
`gpu_memory_utilization` in `{0.6, 0.7}`, and `max_num_seqs=1`. Across all four
completed cells, the planner-level `overestimation_ratio` stayed
`1.7333734577189286` and the planner-level `wasted_fraction` stayed
`0.4230902777777778`. Estimated advisory savings ranged from `685,011,456` to
`1,347,563,520` bytes.

Those values are advisory metrics over planner artifacts. They are not runtime
VRAM measurements, throughput measurements, latency measurements, accuracy
measurements, or quality measurements. The phase deliberately keeps
artifact-input diagnosis separate from controlled substitution.

The same evidence package also records what would be needed before mutation
could be considered. Planner-stage replay matched the runtime scheduler cache
configuration and recorded `runtime_changed_during_replay=false`. Runtime
observation resolved request-to-block assignment, worker tensor layout, and
attention block-table/view metadata. The Mamba side remains blocked: state-index
and state tensor contracts were not safely observable in the current run. This
makes the artifact useful in two ways: it provides a diagnostic CLI users can
run from committed artifacts, and it records why allocator replacement is not
claimed for this phase.

### 1.2 Contributions

The contributions of this report are therefore evidence-bounded:

- a bounded method for capturing and translating vLLM hybrid cache-plan
  artifacts;
- planner-stage replay evidence that matches the runtime scheduler config;
- a 4-cell Path C advisory matrix showing planner-level over-reservation for
  one `Zyphra/Zamba2-2.7B-instruct` setup;
- a deterministic RTX 3060 planner-only comparison showing `cachepawl-avmp`
  stays closer to useful bytes than a uniform padded planner baseline on three
  synthetic hybrid workloads;
- a user-facing `cachepawl diagnose-vllm` artifact-input report that is
  read-only and non-mutating;
- an explicit contract checklist that keeps controlled substitution in future
  work until Mamba state-index and state tensor paths are observable or
  supported upstream.

## 2. Method

The Path C method is an observe/translate/replay/diagnose workflow. It is
designed to preserve the boundary between advisory analysis and runtime
mutation.

### 2.1 Observe Vanilla vLLM Artifacts

The workflow starts from a vanilla `vllm==0.21.0` run for
`Zyphra/Zamba2-2.7B-instruct`. Observation scripts collect cache-plan artifacts
and safe runtime metadata. The observation path is read-only: it does not edit
vLLM source, monkeypatch vLLM internals, replace allocators, or return
Cachepawl plans to vLLM.

The captured artifacts include the runtime cache plan, safe metadata about the
planner inputs, scheduler/KV manager structure, worker tensor layout, live
request/block assignment, and attention-side block-table/view metadata.

### 2.2 Translate Cache Plans

The runtime cache plan is translated into a Cachepawl schema that separates the
fields needed for advisory metrics from the fields that would be required for
future controlled substitution. The translation records cache group count,
cache tensor count, layer count, block count, per-group cache kind, page size,
block size, useful bytes, and observed reserved bytes.

This schema is the input to both the diagnostic CLI and the planner-stage
advisory comparison.

### 2.3 Replay the Planner Stage

The workflow replays `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` on real
planner inputs captured from the observed run. The replay is bounded and
post-initialization. It is used to check whether the planner-stage translation
matches the runtime scheduler cache configuration.

The current evidence records:

- `planner_matches_runtime_scheduler=true`;
- `runtime_changed_during_replay=false`.

This grounds the advisory comparison in the same cache configuration used by
the runtime scheduler, without changing that runtime scheduler state.

### 2.4 Compute Advisory Metrics

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
not imply throughput, latency, accuracy, or quality effects.

### 2.5 Package Artifact-Input Diagnosis

`cachepawl diagnose-vllm` is the productized output of the phase. It consumes an
existing translated cache config and optional raw safe metadata, then writes:

- `report.json`;
- `summary.md`;
- `manifest.json`.

The artifact-input CLI is intentionally lightweight. It does not require vLLM,
GPU access, CUDA, NVML, or model loading. It does not modify vLLM, replace
allocators, or enable runtime mutation.

### 2.6 Inspect Mutation Contracts

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

## 3. Evaluation

The evaluation has two evidence tracks. The Path C track measures
planner-level over-reservation in observed `vllm==0.21.0` cache-plan artifacts
for `Zyphra/Zamba2-2.7B-instruct`. The RTX 3060 planner-comparison track uses a
deterministic synthetic planner harness for `jamba-1.5-mini` workloads. Both
tracks are advisory/planner-level only.

### 3.1 Procedure and Non-Mutation Controls

The Path C workflow observes a vanilla vLLM cache plan, translates it into the
Cachepawl schema, replays `vllm.v1.core.kv_cache_utils.get_kv_cache_configs`
on captured planner inputs, runs `cachepawl diagnose-vllm`, and repeats the
advisory diff over the bounded 4-cell matrix. Planner-stage replay succeeded:

- `planner_matches_runtime_scheduler=true`
- `runtime_changed_during_replay=false`

All observations were read-only. The workflow did not modify vLLM, monkeypatch
private methods, replace allocators, return Cachepawl plans to vLLM, alter
scheduler behavior, or alter worker tensor layout. This validates the advisory
comparison against the observed planner/runtime config, but it does not
establish runtime substitution safety.

The planner-comparison pack is reproduced with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_planner_comparison_pack.py --output-dir benchmarks/results/rtx3060/planner-comparison --seed 1 --num-requests 128 --gpu-name "NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888
```

The pack does not install vLLM, run serving, monkeypatch vLLM, replace
allocators, load a real model, run kernels, or measure latency and throughput.

### 3.2 Metric Definitions

- `reserved_bytes`: bytes reserved by the observed or modeled planner.
- `useful_bytes`: cache payload bytes before planner padding.
- `estimated_savings_bytes = reserved_bytes - useful_bytes` for Path C.
- `overestimation_ratio = reserved_bytes / useful_bytes`.
- `wasted_fraction = (reserved_bytes - useful_bytes) / reserved_bytes`.
- `virtual_oom`: whether a planner estimate exceeds the 12 GiB target profile
  in the synthetic planner pack. It is not an observed runtime OOM.

### 3.3 Path C Advisory Matrix

This table is advisory/diagnostic evidence only. It does not report runtime
mutation, throughput, serving, or VRAM improvement measurements.

| max_model_len | gpu_memory_utilization | reserved bytes | useful bytes | estimated advisory savings bytes |
| ---: | ---: | ---: | ---: | ---: |
| `2048` | `0.6` | `1,893,335,040` | `1,092,283,392` | `801,051,648` |
| `2048` | `0.7` | `3,185,049,600` | `1,837,486,080` | `1,347,563,520` |
| `4096` | `0.6` | `1,619,066,880` | `934,055,424` | `685,011,456` |
| `4096` | `0.7` | `2,910,781,440` | `1,679,258,112` | `1,231,523,328` |

Across all four completed cells, `overestimation_ratio` stayed
`1.7333734577189286` and `wasted_fraction` stayed `0.4230902777777778`.
Estimated advisory savings ranged from `685,011,456` to `1,347,563,520` bytes.
The full matrix, including `num_blocks`, cache group count, cache tensor count,
layer count, and status, is recorded in
`research/avmp/v2/evaluation/matrix_table.md`.

### 3.4 RTX 3060 Planner-Only Comparison

The planner-comparison pack reports deterministic planner estimates for three
synthetic hybrid KV/State workloads. The `vllm-style-padded` backend models a
uniform padded cache planner, not exact vLLM internals. The `cachepawl-avmp`
backend models a native KV-page plus SSM-block planner. The table below is
copied from
`benchmarks/results/rtx3060/planner-comparison/summary.md`. This is planner
evidence for the cache-shape claim, not runtime evidence for vLLM serving
behavior.

| workload | backend | useful bytes | estimated bytes | overestimation | wasted fraction | virtual OOM |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `short-heavy` | `vllm-style-padded` | `2,435,055,616` | `6,985,613,312` | `2.868770` | `0.651418` | `false` |
| `short-heavy` | `cachepawl-avmp` | `2,435,055,616` | `2,451,046,400` | `1.006567` | `0.006524` | `false` |
| `long-heavy` | `vllm-style-padded` | `41,967,550,464` | `165,116,116,992` | `3.934376` | `0.745830` | `true` |
| `long-heavy` | `cachepawl-avmp` | `41,967,550,464` | `41,983,672,320` | `1.000384` | `0.000384` | `true` |
| `mixed` | `vllm-style-padded` | `13,517,422,592` | `51,311,017,984` | `3.795917` | `0.736559` | `true` |
| `mixed` | `cachepawl-avmp` | `13,517,422,592` | `13,532,397,568` | `1.001108` | `0.001107` | `true` |

This supports a narrow planner claim: when the synthetic workload contains
heterogeneous KV and state-cache shapes, AVMP's planner model keeps estimated
bytes close to useful bytes, while the uniform padded planner model reserves
substantially more bytes. The result does not prove live serving memory
reduction or request-admission improvement. The pack does not run vLLM serving,
and `long-heavy` plus `mixed` remain virtual-OOM even under AVMP because useful
demand itself exceeds the 12 GiB target profile.

### 3.5 Runtime Contracts and Claim Boundary

The runtime contract observations resolved several mutation-readiness questions
on the observation side:

- Scheduler/KVCacheManager structure was observed.
- Block usage metadata was observed.
- Worker tensor layout was observed. The runtime contract artifact captured 32
  worker tensor summaries; the first had shape `[2, 329, 48, 32, 160]`.
- Live request-to-block assignment was observed. A bounded request saw active
  block ids `[1, 2, 3, 4, 5, 6, 7]` after the first scheduler step.
- Attention block-table/view metadata was observed, with 21 block-table tensor
  metadata summaries.
- Attention metadata builders were observed, with 7 attention groups.

The Mamba side remains gated:

- `mamba_state_index_contract` is blocked because `mamba_state_idx` was
  reachable but empty for the live request.
- `mamba_state_tensor_contract` is blocked because no Mamba state tensors were
  safely reachable by stable runtime attributes.
- Runtime cache config reported `mamba_cache_mode: none`.

These blockers are decisive for this phase. Without Mamba state-index and state
tensor contracts, the evidence supports observe/advisory reporting only.

The supported claims are the advisory CLI, planner-stage replay match, bounded
Path C over-reservation matrix, read-only observation, and synthetic
planner-only AVMP comparison. The evidence does not show runtime allocator
replacement, runtime cache substitution, measured runtime VRAM reduction,
throughput improvement, latency improvement, quality improvement, or accuracy
improvement.

## 4. Limitations

This report is scoped to a diagnostic/advisory systems artifact. Its main
limitations are:

- no runtime mutation: the evaluation does not replace vLLM allocators, rewrite
  cache views, alter scheduler behavior, alter worker tensor layout, or return
  Cachepawl plans to vLLM;
- no serving performance or quality measurement: the evidence does not include
  runtime VRAM reduction, throughput, latency, serving, quality, or accuracy
  experiments;
- narrow Path C scope: one observed model, vanilla `vllm==0.21.0`,
  `max_model_len` in `{2048, 4096}`, `gpu_memory_utilization` in `{0.6, 0.7}`,
  and `max_num_seqs=1`;
- narrow planner-comparison scope: synthetic `jamba-1.5-mini` workloads and a
  deterministic RTX 3060 target profile, not real model inference;
- local platform scope: the Path C artifacts are bounded to the recorded local
  RTX 3060 12 GiB / WSL2 context and should not be generalized to production
  serving clusters.

The Mamba-side mutation contracts remain blocked:

- `mamba_state_index_contract`, because `mamba_state_idx` was reachable but
  empty for the live request;
- `mamba_state_tensor_contract`, because no Mamba state tensors were safely
  reachable by stable runtime attributes.

The observed runtime cache config reported `mamba_cache_mode: none`, so this
run did not populate the Mamba state-index/state-tensor paths needed for a view
rewrite contract. Because these contracts are blocked, the report cannot claim
controlled substitution readiness.

Future work must first improve Mamba state-index observability. A controlled
substitution experiment should only be considered after one of the following is
available:

- a run/model/config where Mamba state-index and state tensors are observable
  through bounded read-only runtime paths;
- a supported vLLM integration seam for external cache allocators or view
  rewrites;
- a default-off mutation probe with explicit rollback controls and validated
  Mamba/attention rewrite contracts.

The next evaluation step is multi-model and multi-workload advisory evaluation,
still without conflating advisory savings with measured runtime improvements.

## 5. Conclusion

Cachepawl Path C supports a narrow but useful systems claim: planner-level
hybrid cache over-reservation can be diagnosed from committed vLLM cache-plan
artifacts without modifying the serving stack. The current evidence includes a
bounded `Zyphra/Zamba2-2.7B-instruct` Path C matrix, a deterministic RTX 3060
planner-only comparison, and an artifact-input diagnostic CLI. Together these
support advisory reporting, not runtime replacement.

The technical ambition remains controlled substitution for hybrid cache
planning, but that is future work. Before this paper can claim live VRAM
reduction, throughput, latency, quality, accuracy, or allocator replacement, it
needs Mamba state-index and state tensor contracts, a default-off substitution
probe, and live serving measurements with rollback and parity controls.

## 6. Artifact and Reproducibility Checklist

All paths are relative to the repository root. This checklist maps each report
claim to committed evidence and records the commands needed to verify the
venue-neutral Markdown package.

### 6.1 Diagnostic CLI Outputs

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json` | User-facing advisory report with planner-level reserved/useful/savings metrics. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/summary.md` | Human-readable advisory summary and non-mutation statement. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/manifest.json` | Artifact-input mode, output list, and non-requirement flags for vLLM, CUDA, GPU, and NVML. |

### 6.2 Planner Replay and Advisory Diff

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json` | Planner-stage replay output translated into Cachepawl's schema. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json` | Planner-stage advisory metrics and mutation-prevention field list. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json` | Per-cache-group advisory diff details. |

Supported claims:

- planner-stage replay matched the runtime scheduler cache config;
- `runtime_changed_during_replay=false`;
- the advisory comparison is grounded in the observed planner/runtime config.

### 6.3 Four-Cell Advisory Matrix

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-path-c-matrix/` | Four bounded planner-stage advisory matrix cells. |
| `research/avmp/v2/evaluation/matrix_table.csv` | Deterministic machine-readable consolidated matrix metrics. |
| `research/avmp/v2/evaluation/matrix_table.md` | Paper-readable consolidated matrix metrics. |

Matrix scope:

- model: `Zyphra/Zamba2-2.7B-instruct`;
- vLLM version: `0.21.0`;
- `max_model_len` in `{2048, 4096}`;
- `gpu_memory_utilization` in `{0.6, 0.7}`;
- `max_num_seqs=1`.

Matrix claims:

- `overestimation_ratio=1.7333734577189286`;
- `wasted_fraction=0.4230902777777778`;
- advisory savings range from `685,011,456` to `1,347,563,520` bytes.

These are planner-level advisory metrics. They are not runtime VRAM reduction
measurements.

### 6.4 RTX 3060 Planner-Comparison Pack

| Artifact | Claim Supported |
| --- | --- |
| `benchmarks/results/rtx3060/planner-comparison/README.md` | Reproduction command, planner-only interpretation, target GPU profile, and exclusions. |
| `benchmarks/results/rtx3060/planner-comparison/summary.md` | Compact planner result table for `vllm-style-padded` and `cachepawl-avmp`. |
| `benchmarks/results/rtx3060/planner-comparison/manifest.json` | Seed, request count, target GPU bytes, schema version, and generation command. |
| `benchmarks/results/rtx3060/planner-comparison/environment.json` | Captured environment and target GPU metadata. |
| `benchmarks/results/rtx3060/planner-comparison/short-heavy.jsonl` | Per-backend planner records for the short-heavy workload. |
| `benchmarks/results/rtx3060/planner-comparison/long-heavy.jsonl` | Per-backend planner records for the long-heavy workload. |
| `benchmarks/results/rtx3060/planner-comparison/mixed.jsonl` | Per-backend planner records for the mixed workload. |

Planner-comparison claims:

- the result is deterministic at seed `1` with runtime measurement disabled;
- `cachepawl-avmp` estimated bytes remain close to useful bytes for the three
  synthetic workloads;
- the uniform padded planner baseline has higher overestimation and wasted
  fraction in the same workload records.

These are planner-only metrics. They do not report runtime vLLM serving,
allocator replacement, live VRAM savings, throughput, latency, or quality.

### 6.5 Runtime Contract Observations

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-runtime-contract-observation/runtime_contract_report.json` | Scheduler/KV manager structure, block usage metadata, and worker tensor layout observation. |
| `research/avmp/v2/results/vllm-live-request-contract-observation/live_request_contract_report.json` | Live request id, request-to-block assignment, and block-pool snapshots. |
| `research/avmp/v2/results/vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` | Attention block-table/view metadata and remaining Mamba blockers. |
| `research/avmp/v2/results/vllm-mutation-readiness/readiness_report.json` | Advisory-only recommendation and structural mutation-readiness checks. |

Observed contracts:

- request-to-block assignment;
- worker tensor layout;
- attention block-table/view metadata.

Blocked contracts:

- Mamba state-index contract;
- Mamba state tensor contract.

### 6.6 Phase Interpretation

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/PATH_C_OBSERVE_ADVISORY_PHASE_REPORT.md` | Cycle-close interpretation and advisory-only recommendation. |
| `research/avmp/v2/evaluation/README.md` | Consolidated scope, matrix result, and limitation summary. |
| `research/avmp/v2/evaluation/artifact_index.md` | Cross-artifact claim map for the evidence pack. |
| `research/avmp/v2/evaluation/claim_summary.md` | Paper-facing claim boundary, result statement, and unsupported claims. |
| `research/avmp/v2/results/vllm-baseline/manifest.json` | Local RTX 3060 12 GiB and WSL2 environment context for the bounded Path C work. |

The artifact set supports observe/advisory diagnosis only. It does not support
claims about runtime allocator replacement, runtime VRAM reduction, latency,
throughput, accuracy, or model quality.

### 6.7 Reproducibility Checklist

| Item | Status | Evidence or Command |
| --- | --- | --- |
| Path C claim boundary documented | Ready | `research/avmp/v2/evaluation/claim_summary.md` |
| Path C matrix table committed | Ready | `research/avmp/v2/evaluation/matrix_table.md`, `matrix_table.csv` |
| Planner-comparison pack committed | Ready | `benchmarks/results/rtx3060/planner-comparison/` |
| Planner-comparison pack regeneration | Ready | `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_planner_comparison_pack.py --output-dir benchmarks/results/rtx3060/planner-comparison --seed 1 --num-requests 128 --gpu-name "NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888` |
| Diagnostic CLI tests | Ready | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/cli/test_diagnose_vllm.py -q` |
| Planner evidence tests | Ready | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_planner_comparison.py tests/bench/test_vllm_path_c_advisory_matrix.py -q` |
| Artifact path check | Ready | Confirm every path listed in this section exists in the repository. |
| Venue-specific formatting | TODO | Select target template, convert Markdown to venue format, add required author metadata and bibliography style. |

### 6.8 Submission Readiness

Ready:

- venue-neutral Markdown draft with title, abstract, introduction, method,
  evaluation, limitations, conclusion, artifact map, and reproducibility
  checklist;
- compact evaluation tables for the Path C matrix and RTX 3060 planner-only
  comparison;
- explicit advisory/planner-level claim boundary throughout the paper;
- committed evidence paths for numeric claims and product artifacts.

Remaining before venue submission:

- choose the target venue format and page limit;
- convert Markdown sections into the target template;
- add venue-required author, anonymization, ethics, artifact, and bibliography
  metadata;
- decide whether to keep the full artifact checklist in the main paper,
  appendix, or supplemental material;
- run the venue-specific build once the target template exists.
