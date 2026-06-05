# vLLM Path C Observe/Advisory Evaluation Pack

This directory consolidates the completed vLLM Path C observe/advisory evidence
for paper and product use.

Conclusion: this is advisory/diagnostic evidence, not runtime mutation
evidence. The `cachepawl diagnose-vllm` artifact-input path is valid and useful,
but controlled substitution is not approved for this cycle.

Runtime validation update: the sprint outcome is `blocker`. Stock vLLM runtime
evidence exists for the bounded hybrid-model baseline, planner/admission replay
exists, and controlled substitution remains blocked by missing stable Mamba
state-index/state tensor contracts. The 2026-06-05 rerun could not refresh the
live baseline because CUDA/NVML access is currently unavailable.

Runtime proof sprint update: the strongest achievable tier is now
`partial_success`. The new proof pack exhausts local CUDA-gated scenarios,
prepares exact GPU-machine commands for prefix-caching and Mamba cache modes,
and records a minimal Mamba state contract proposal. It still does not claim
runtime AVMP substitution or runtime savings.

## Scope

- Model: `Zyphra/Zamba2-2.7B-instruct`
- vLLM version: `0.21.0`
- Python executable: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python`
- Durable vLLM environment: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv`
- Max model lengths: `2048`, `4096`
- Max number of sequences: `1`
- GPU memory utilization values: `0.6`, `0.7`
- Observed tensor device: `cuda:0`

## Files

- `metrics_table.md` - paper-readable metrics table.
- `metrics_table.csv` - deterministic machine-readable metrics table.
- `matrix_table.md` - 4-cell advisory matrix table.
- `matrix_table.csv` - deterministic machine-readable matrix table.
- `config_matrix.json` - bounded matrix definition.
- `artifact_index.md` - source artifact index and what each artifact supports.
- `claim_summary.md` - paper-facing claim, evidence, and limitation summary.
- `methodology.md` - bounded observation and advisory methodology.
- `limitations.md` - limitations and mutation blockers.
- `../results/vllm-runtime-validation-sprint/` - runtime validation sprint
  outcome, current rerun blocker, and substitution blocker report.
- `../results/vllm-runtime-proof-sprint/` - expanded scenario matrix,
  GPU-machine commands, local blocked runs, and Mamba state contract proposal.

## Headline Result

The planner-stage advisory matrix now covers four bounded configuration cells:
`max_model_len` in `{2048, 4096}`, `gpu_memory_utilization` in `{0.6, 0.7}`,
and `max_num_seqs=1` for the same model. Across all completed cells, the
overestimation ratio stayed `1.7333734577189286` and the wasted fraction stayed
`0.4230902777777778`.

Estimated advisory savings varied by cell:

- `2048 / 0.6 / 1`: `801,051,648` bytes
- `2048 / 0.7 / 1`: `1,347,563,520` bytes
- `4096 / 0.6 / 1`: `685,011,456` bytes
- `4096 / 0.7 / 1`: `1,231,523,328` bytes

These are planner-level advisory savings, not measured runtime VRAM reductions.

The runtime evidence resolved request-to-block assignment, worker tensor layout,
attention block-table/view metadata, and attention metadata builders. Mamba
state-index and Mamba state tensor contracts remain blocked for this run.
