# vLLM Runtime Proof Sprint

Best success tier achieved: `partial_success`

This sprint tried to push beyond the previous blocker result toward positive
AVMP/vLLM runtime evidence. No paper polish was performed. The work stayed in
runtime artifacts, scenario matrices, and a minimal missing-contract proposal.

## Summary

The strongest positive evidence remains a live stock vLLM runtime observation
plus planner/admission replay:

- Stock vLLM completed bounded generation on
  `Zyphra/Zamba2-2.7B-instruct`.
- Live request admission and request-to-block assignment were observed.
- Block-pool usage changed during the request and recovered after completion.
- Planner replay measured the reserved/useful byte gap and a Cachepawl
  advisory plan beside the vanilla plan.
- Attention block tables and attention metadata groups were observed.

Direct AVMP/default-off substitution did not run. The committed live run did
not expose Mamba state-index or Mamba state tensor contracts, and the current
local environment cannot execute new CUDA/NVML runtime scenarios.

## Local Environment Result

The pinned vLLM environment is present and importable:

- Python: `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python`
- vLLM: `0.21.0`
- CUDA reported by torch package: `13.0`
- `torch.cuda.is_available()`: `false`
- CUDA device count: `0`
- `nvidia-smi`: failed with GPU access blocked by the operating system

This makes local cache-mode and prefix-caching experiments
environment-blocked, not research-blocked.

## Scenarios Tried Or Prepared

See:

- `scenario_matrix.json`
- `scenario_matrix.md`
- `local-blocked-runs/`
- `gpu_machine_commands.sh`

Local blocked captures were run for:

- `max_model_len=1024`, `max_num_seqs=1`, short prompt
- `max_model_len=2048`, `max_num_seqs=2`, longer prompt
- live-request contract probe at `max_model_len=1024`
- Mamba/attention contract probe at `max_model_len=1024`

All local fresh runtime captures stopped at the same CUDA/NVML gate.

GPU-machine commands were prepared for:

- prefix caching on and off
- `mamba_cache_mode=none`, `align`, and `all`
- `max_model_len=1024` and `2048`
- `max_num_seqs=1` and `4`
- short and longer prompts

## Model Matrix Result

The only locally cached hybrid candidate is
`Zyphra/Zamba2-2.7B-instruct`. Its Hugging Face config identifies:

- `architecture=Zamba2ForCausalLM`
- `model_type=zamba2`
- `num_hidden_layers=54`
- `mamba_d_state=64`
- `mamba_expand=2`

`Qwen/Qwen2.5-1.5B-Instruct` is cached but transformer-only
(`Qwen2ForCausalLM`, `model_type=qwen2`) and was rejected as positive hybrid
evidence. Other hybrid/SSM candidates were not cached and were not downloaded
because local CUDA is unavailable.

## Version Matrix Result

The installed pinned version is `vllm==0.21.0`. PyPI reported `0.21.0` as the
latest stable vLLM release on 2026-06-05, so creating a separate latest-stable
environment would not change the tested version. Source/nightly was not
attempted because stable cannot reach CUDA locally.

## Substitution Result

No substitution was attempted. The gate remains:

- Mamba state-index contract must be observable.
- Mamba state tensor contract must be observable.
- Output parity and rollback must be available before any mutation.

The proposal in `vllm_mamba_state_contract_proposal.md` defines the minimal
contract that vLLM or a local fork should expose before Cachepawl attempts a
default-off micro-substitution.

## Claim Boundary

This sprint does not claim:

- AVMP runtime memory reduction
- request-admission improvement
- latency improvement
- throughput improvement
- quality or accuracy improvement
- successful runtime substitution

It does support a partial positive result: live stock vLLM runtime and
admission behavior are connected to a measured planner gap, with exact
CUDA-required scenarios and a concrete Mamba state contract seam for the next
runtime proof step.

## Files

- `README.md`: this summary
- `scenario_matrix.json`: machine-readable scenario outcomes
- `scenario_matrix.md`: reviewer-readable scenario matrix
- `gpu_machine_commands.sh`: exact GPU-host runtime commands
- `vllm_mamba_state_contract_proposal.md`: minimal missing-contract proposal
- `local-blocked-runs/`: raw local blocker artifacts from fresh scenario runs
