# vLLM Live Request Contract Observation

Status: `live_request_contract_observation_complete`

This artifact captures bounded read-only live-request metadata from vanilla
`vllm==0.21.0` for `Zyphra/Zamba2-2.7B-instruct`.

Files:

- `manifest.json` - capture status, parameters, outputs, and non-mutation flags.
- `live_request_contract_report.json` - live request id, request/block snapshots,
  scheduler request metadata, and field status records.
- `raw_safe_metadata.json` - scalar runtime object metadata only.
- `field_level_blockers.json` - fields not safely observable in this bounded run.
- `summary.md` - concise human-readable summary.
