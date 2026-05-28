# Artifact Appendix

All paths are relative to the repository root. The appendix maps each report
claim to committed evidence.

## Diagnostic CLI Outputs

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json` | User-facing advisory report with planner-level reserved/useful/savings metrics. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/summary.md` | Human-readable advisory summary and non-mutation statement. |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/manifest.json` | Artifact-input mode, output list, and non-requirement flags for vLLM, CUDA, GPU, and NVML. |

## Planner Replay and Advisory Diff

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json` | Planner-stage replay output translated into Cachepawl's schema. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json` | Planner-stage advisory metrics and mutation-prevention field list. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json` | Per-cache-group advisory diff details. |

Supported claims:

- planner-stage replay matched the runtime scheduler cache config;
- `runtime_changed_during_replay=false`;
- the advisory comparison is grounded in the observed planner/runtime config.

## Four-Cell Advisory Matrix

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

## Runtime Contract Observations

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

## Phase Interpretation

| Artifact | Claim Supported |
| --- | --- |
| `research/avmp/v2/PATH_C_OBSERVE_ADVISORY_PHASE_REPORT.md` | Cycle-close interpretation and advisory-only recommendation. |
| `research/avmp/v2/evaluation/README.md` | Consolidated scope, matrix result, and limitation summary. |
| `research/avmp/v2/evaluation/artifact_index.md` | Cross-artifact claim map for the evidence pack. |
| `research/avmp/v2/results/vllm-baseline/manifest.json` | Local RTX 3060 12 GiB and WSL2 environment context for the bounded Path C work. |

The artifact set supports observe/advisory diagnosis only. It does not support
claims about runtime allocator replacement, runtime VRAM reduction, latency,
throughput, or model quality.
