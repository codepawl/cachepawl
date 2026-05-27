# vLLM Path C Observe/Advisory Evaluation Pack

This directory consolidates the completed vLLM Path C observe/advisory evidence
for paper and product use.

Conclusion: this is advisory/diagnostic evidence, not runtime mutation
evidence. The `cachepawl diagnose-vllm` artifact-input path is valid and useful,
but controlled substitution is not approved for this cycle.

## Scope

- Model: `Zyphra/Zamba2-2.7B-instruct`
- vLLM version: `0.21.0`
- Python executable: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python`
- Durable vLLM environment: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- Max model length: `4096`
- Max number of sequences: `1`
- GPU memory utilization: `0.7`
- Observed tensor device: `cuda:0`

## Files

- `metrics_table.md` - paper-readable metrics table.
- `metrics_table.csv` - deterministic machine-readable metrics table.
- `artifact_index.md` - source artifact index and what each artifact supports.
- `methodology.md` - bounded observation and advisory methodology.
- `limitations.md` - limitations and mutation blockers.

## Headline Result

The planner-stage advisory estimates that vanilla vLLM reserves
`2,910,781,440` bytes for the observed cache plan while the useful bytes are
`1,679,258,112`. Cachepawl's advisory plan therefore reports
`1,231,523,328` bytes of estimated savings, an overestimation ratio of
`1.7333734577189286`, and a wasted fraction of `0.4230902777777778`.

The runtime evidence resolved request-to-block assignment, worker tensor layout,
attention block-table/view metadata, and attention metadata builders. Mamba
state-index and Mamba state tensor contracts remain blocked for this run.
