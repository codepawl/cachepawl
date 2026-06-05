# Artifact Index

All paths are relative to the repository root.

| Artifact | Role |
| --- | --- |
| `research/avmp/v2/results/vllm-runtime-cache-diagnostic-cli/report.json` | User-facing `cachepawl diagnose-vllm` advisory report. |
| `research/avmp/v2/results/vllm-planner-stage-observation/translated_planner_stage_config.json` | Planner-stage replay output translated into Cachepawl's vLLM cache-plan schema. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/diff_report.json` | Main advisory metrics, parity flags, and mutation-prevention field list. |
| `research/avmp/v2/results/vllm-planner-stage-advisory-diff/group_level_diff.json` | Per-cache-group advisory diff details. |
| `research/avmp/v2/results/vllm-mutation-readiness/readiness_report.json` | Structural mutation-readiness checks and advisory-only classification. |
| `research/avmp/v2/results/vllm-runtime-contract-observation/runtime_contract_report.json` | Scheduler/KV manager, block usage, worker tensor layout, and initial contract blockers. |
| `research/avmp/v2/results/vllm-live-request-contract-observation/live_request_contract_report.json` | Live request id, request-to-block assignment, scheduler request metadata, and block-pool before/after snapshots. |
| `research/avmp/v2/results/vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` | Attention block-table/view metadata, attention group metadata, and remaining Mamba state blockers. |
| `research/avmp/v2/results/vllm-runtime-validation-sprint/README.md` | Runtime validation sprint outcome, stock vLLM baseline summary, current rerun blocker, planner replay, live admission evidence, and substitution blocker interpretation. |
| `research/avmp/v2/results/vllm-runtime-validation-sprint/outcome.json` | Machine-readable sprint outcome and claim boundary. |
| `research/avmp/v2/results/vllm-runtime-validation-sprint/current_baseline_blocker_manifest.json` | Current 2026-06-05 rerun blocker showing pinned vLLM is present but CUDA/NVML is unavailable. |
| `research/avmp/v2/results/vllm-runtime-proof-sprint/scenario_matrix.json` | Expanded config/model/version/probe matrix, local environment blockers, and prepared GPU-machine scenarios. |
| `research/avmp/v2/results/vllm-runtime-proof-sprint/gpu_machine_commands.sh` | Exact GPU-host commands for prefix-caching and `mamba_cache_mode` variants. |
| `research/avmp/v2/results/vllm-runtime-proof-sprint/vllm_mamba_state_contract_proposal.md` | Minimal upstream/local-fork contract proposal for exposing Mamba state-index and state tensor summaries. |
| `research/avmp/v2/results/vllm-path-c-matrix/` | Four bounded planner-stage advisory matrix cells for `max_model_len` in `{2048, 4096}` and `gpu_memory_utilization` in `{0.6, 0.7}`. |
| `research/avmp/v2/evaluation/matrix_table.csv` | Deterministic consolidated matrix metrics table. |
| `research/avmp/v2/evaluation/matrix_table.md` | Paper-readable consolidated matrix metrics table. |
| `research/avmp/v2/PATH_C_OBSERVE_ADVISORY_PHASE_REPORT.md` | Cycle-closure interpretation and advisory-only recommendation. |

## Cross-Artifact Claims

- Planner-stage replay matched the runtime scheduler cache config:
  `planner_matches_runtime_scheduler=true`.
- Replay did not alter runtime scheduler state:
  `runtime_changed_during_replay=false`.
- `cachepawl diagnose-vllm` reports the same headline advisory metrics as the
  planner-stage diff.
- The 4-cell advisory matrix completed for one model and one sequence. Across
  completed cells, `overestimation_ratio=1.7333734577189286` and
  `wasted_fraction=0.4230902777777778`; estimated advisory savings ranged from
  `685,011,456` to `1,347,563,520` bytes.
- Live runtime observations are read-only and bounded. They do not modify vLLM,
  monkeypatch private methods, replace allocators, alter scheduler behavior, or
  return Cachepawl plans to vLLM.
- The runtime validation sprint outcome is `blocker`: stock vLLM runtime
  evidence exists, planner/admission replay exists, and controlled substitution
  remains blocked by missing stable Mamba state-index/state tensor contracts.
- The runtime proof sprint outcome is `partial_success`: positive stock vLLM
  runtime evidence is connected to planner/admission replay, but new local
  cache-mode scenarios are CUDA/NVML-blocked and substitution still requires
  observable Mamba state contracts.
