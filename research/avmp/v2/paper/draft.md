# Cachepawl Path C: Advisory Diagnosis of Hybrid Attention/Mamba Cache Overestimation in vLLM

## Abstract

Hybrid Attention/Mamba language models stress inference cache planners because
attention KV pages and Mamba state caches have different shapes and reuse
contracts. Cachepawl Path C evaluates whether these planner-level inefficiencies
can be diagnosed in a real vLLM runtime without modifying the serving stack.

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

The artifact also records runtime contracts needed before any mutation attempt.
Live request-to-block assignment, worker tensor layout, and attention
block-table/view metadata were observed. Mamba state-index and Mamba state
tensor contracts remain blocked: `mamba_state_idx` was reachable but empty for
the live request, no Mamba state tensors were safely reachable, and the observed
cache config used `mamba_cache_mode: none`.

This is a diagnostic/advisory systems artifact. It does not claim runtime cache
substitution, allocator replacement, serving-time VRAM reduction, throughput
improvement, latency improvement, accuracy improvement, or quality impact. Its
contribution is a non-invasive method and product surface for finding hybrid
cache overestimation while preserving explicit gates for future controlled
substitution.

## 1. Introduction

Hybrid Attention/Mamba language models place two different cache shapes behind
one inference runtime. Attention layers use KV blocks whose economics are tied
to token sequence growth, while Mamba or SSM layers use state tensors whose
useful byte footprint can diverge from the page shape used by a uniform cache
planner. A cache planner that reserves by a shared page/block abstraction can
therefore allocate capacity according to a shape that is larger than the useful
state payload for hybrid layouts.

Cachepawl Path C studies this gap as an observe/advisory problem. The goal of
this phase is not to replace vLLM's allocator or to rewrite live runtime state.
Instead, the goal is to answer a narrower systems question: can Cachepawl
observe a real vanilla vLLM hybrid cache plan, translate it into a stable
schema, replay the planner stage on the observed inputs, and emit a practical
diagnostic report that quantifies planner-level overestimation?

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

The contribution of this report is therefore an observe/advisory systems
artifact:

- a bounded method for capturing and translating vLLM hybrid cache-plan
  artifacts;
- planner-stage replay evidence that matches the runtime scheduler config;
- a user-facing `cachepawl diagnose-vllm` artifact-input report;
- a 4-cell advisory matrix with stable overestimation and waste metrics;
- an explicit contract checklist that blocks runtime mutation until Mamba
  state-index and state tensor paths are observable or supported upstream.

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

### 3.1 Setup

The evaluation uses existing observe/advisory artifacts from a vanilla
`vllm==0.21.0` run of `Zyphra/Zamba2-2.7B-instruct`.

Configuration:

- Max model lengths: `2048`, `4096`
- Max number of sequences: `1`
- GPU memory utilization values: `0.6`, `0.7`
- Durable vLLM environment:
  `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- Observed tensor device: `cuda:0`
- Reference local platform for the vLLM Path C work: RTX 3060 12 GiB under
  WSL2, as recorded by the surrounding baseline and setup artifacts.

The matrix artifacts themselves are interpreted as planner-level advisory
artifacts. The local hardware and WSL2 context are reported as an environment
limit, not as a basis for throughput, latency, or runtime memory claims.

The artifacts are used only for planner-level advisory analysis and runtime
contract observation. They are not used to claim runtime VRAM reduction,
throughput improvement, latency improvement, model quality impact, or allocator
replacement.

### 3.2 Evaluation Procedure

The evaluation has five stages:

1. Observe the vanilla vLLM runtime cache plan and safe runtime metadata.
2. Translate the vLLM cache plan into Cachepawl's schema.
3. Replay `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` on real planner
   inputs in a bounded post-initialization run.
4. Run `cachepawl diagnose-vllm` on artifact inputs to produce advisory metrics.
5. Repeat the planner-stage observation and advisory diff over the bounded
   4-cell advisory matrix.

All runtime observations were read-only. The workflow did not modify vLLM,
monkeypatch private methods, replace allocators, return Cachepawl plans to vLLM,
alter scheduler behavior, or alter worker tensor layout.

### 3.3 Planner Replay Result

Planner-stage replay succeeded and produced `planner_stage_translation`. The
translated planner output matched the runtime scheduler cache config:

- `planner_matches_runtime_scheduler=true`
- `runtime_changed_during_replay=false`

This establishes that the advisory comparison is grounded in the same cache
configuration used by the runtime scheduler for each observed planner-stage
matrix cell. It does not establish that Cachepawl can safely substitute that
configuration at runtime.

### 3.4 Advisory Matrix

This table is advisory/diagnostic evidence only. It does not report runtime
mutation, throughput, serving, or VRAM improvement measurements.

| model | max_model_len | gpu_memory_utilization | max_num_seqs | vanilla_reserved_bytes | vanilla_useful_bytes | cachepawl_proposed_reserved_bytes | estimated_savings_bytes | overestimation_ratio | wasted_fraction | num_blocks | cache_group_count | cache_tensor_count | layer_count | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.6 | 1 | 1893335040 | 1092283392 | 1092283392 | 801051648 | 1.7333734577189286 | 0.4230902777777778 | 214 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 2048 | 0.7 | 1 | 3185049600 | 1837486080 | 1837486080 | 1347563520 | 1.7333734577189286 | 0.4230902777777778 | 360 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.6 | 1 | 1619066880 | 934055424 | 934055424 | 685011456 | 1.7333734577189286 | 0.4230902777777778 | 183 | 7 | 9 | 63 | completed |  |
| Zyphra/Zamba2-2.7B-instruct | 4096 | 0.7 | 1 | 2910781440 | 1679258112 | 1679258112 | 1231523328 | 1.7333734577189286 | 0.4230902777777778 | 329 | 7 | 9 | 63 | completed_existing_baseline |  |

Across all four completed cells, `overestimation_ratio` stayed
`1.7333734577189286` and `wasted_fraction` stayed `0.4230902777777778`.
Estimated advisory savings ranged from `685,011,456` to `1,347,563,520` bytes.

The estimated savings are planner-level advisory savings. They are not measured
runtime VRAM reductions, and they do not support latency, throughput, accuracy,
or quality claims.

### 3.5 Runtime Contract Observations

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

### 3.6 Product Artifact

`cachepawl diagnose-vllm` is the productized output of this phase. It consumes
translated cache-plan artifacts and emits `report.json`, `summary.md`, and
`manifest.json`. It is designed for advisory use in environments that do not
have vLLM, CUDA, GPU access, or NVML installed.

### 3.7 Interpretation

The evaluation supports a diagnostic claim: Cachepawl can expose and report
planner-level cache overestimation for a real hybrid vLLM cache plan across a
small bounded config matrix. It remains a single-model advisory result. It does
not support a mutation claim. Controlled substitution requires stronger Mamba
state-index and state tensor contracts, or a supported upstream integration
path.

## 4. Limitations

This report is scoped to a diagnostic/advisory systems artifact. The limitations
below are part of the claim boundary.

### 4.1 No Runtime Mutation Claim

The evaluation does not replace vLLM allocators, rewrite cache views, alter
scheduler behavior, alter worker tensor layout, or return Cachepawl plans to
vLLM. The reported savings are estimated advisory savings from planner
artifacts, not measured serving-time memory savings.

### 4.2 No Performance Or Quality Claim

The evidence does not include:

- runtime VRAM reduction measurements,
- throughput measurements,
- latency measurements,
- serving experiments,
- model quality or accuracy evaluation.

The 4-cell matrix supports only planner-level advisory interpretation.

### 4.3 Single Observed Model

The current evidence is from one model and four bounded config cells:

- `Zyphra/Zamba2-2.7B-instruct`
- vanilla `vllm==0.21.0`
- `max_model_len` in `{2048, 4096}`
- `max_num_seqs=1`
- `gpu_memory_utilization` in `{0.6, 0.7}`

The result should not be generalized to all vLLM versions, models, backends,
cache modes, or workloads without further evaluation.

### 4.4 Local Hardware And Platform

The Path C work was developed and observed on the local RTX 3060 12 GiB / WSL2
environment recorded by the surrounding baseline and setup artifacts. This
environment is sufficient for the bounded observe/advisory evidence, but it is
not a broad hardware study. The report does not claim behavior on other GPUs,
other operating environments, non-WSL2 Linux deployments, or production serving
clusters.

### 4.5 Mamba State Contract Blocker

The remaining blocker is Mamba state observability. The Mamba/attention
contract observation resolved attention block-table/view metadata and attention
metadata builders, but did not resolve:

- `mamba_state_index_contract`, because `mamba_state_idx` was reachable but
  empty for the live request;
- `mamba_state_tensor_contract`, because no Mamba state tensors were safely
  reachable by stable runtime attributes.

The observed runtime cache config reported `mamba_cache_mode: none`, so this
run did not populate the Mamba state-index/state-tensor paths needed for a
rewrite contract.

Because these contracts are blocked, the report cannot claim controlled
substitution readiness.

### 4.6 Future Work

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

## 5. Artifact Appendix

All paths are relative to the repository root. The appendix maps each report
claim to committed evidence.

### 5.1 Diagnostic CLI Outputs

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json` | User-facing advisory report with planner-level reserved/useful/savings metrics. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/summary.md` | Human-readable advisory summary and non-mutation statement. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/manifest.json` | Artifact-input mode, output list, and non-requirement flags for vLLM, CUDA, GPU, and NVML. |

### 5.2 Planner Replay and Advisory Diff

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json` | Planner-stage replay output translated into Cachepawl's schema. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json` | Planner-stage advisory metrics and mutation-prevention field list. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json` | Per-cache-group advisory diff details. |

Supported claims:

- planner-stage replay matched the runtime scheduler cache config;
- `runtime_changed_during_replay=false`;
- the advisory comparison is grounded in the observed planner/runtime config.

### 5.3 Four-Cell Advisory Matrix

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

### 5.4 Runtime Contract Observations

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

### 5.5 Phase Interpretation

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
