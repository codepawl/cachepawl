# vLLM Runtime Contract Observation

Status: `runtime_contract_observation_with_field_blockers`

This artifact captures read-only runtime contract metadata from vanilla
`vllm==0.21.0` for `Zyphra/Zamba2-2.7B-instruct`.

Files:

- `manifest.json` — capture status, parameters, outputs, and non-mutation flags.
- `runtime_contract_report.json` — scheduler/manager, block usage, worker tensor,
  Mamba/attention, and field status records.
- `raw_safe_metadata.json` — scalar runtime object metadata only.
- `field_level_blockers.json` — fields not safely observable in this bounded run.
- `summary.md` — concise human-readable summary.
