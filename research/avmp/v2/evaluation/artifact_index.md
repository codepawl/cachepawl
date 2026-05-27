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
| `research/avmp/v2/PATH_C_OBSERVE_ADVISORY_PHASE_REPORT.md` | Cycle-closure interpretation and advisory-only recommendation. |

## Cross-Artifact Claims

- Planner-stage replay matched the runtime scheduler cache config:
  `planner_matches_runtime_scheduler=true`.
- Replay did not alter runtime scheduler state:
  `runtime_changed_during_replay=false`.
- `cachepawl diagnose-vllm` reports the same headline advisory metrics as the
  planner-stage diff.
- Live runtime observations are read-only and bounded. They do not modify vLLM,
  monkeypatch private methods, replace allocators, alter scheduler behavior, or
  return Cachepawl plans to vLLM.
