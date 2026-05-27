# vLLM Mamba Attention Contract Observation

Status: `mamba_attention_contract_observation_with_field_blockers`

This artifact captures bounded read-only Mamba state-index, attention group, and
block-table metadata from vanilla `vllm==0.21.0` for
`Zyphra/Zamba2-2.7B-instruct`.

Files:

- `manifest.json` - capture status, parameters, outputs, and non-mutation flags.
- `mamba_attention_contract_report.json` - safe runtime metadata summaries.
- `raw_safe_metadata.json` - scalar runtime object metadata only.
- `field_level_blockers.json` - fields not safely observable in this bounded run.
- `summary.md` - concise human-readable summary.
