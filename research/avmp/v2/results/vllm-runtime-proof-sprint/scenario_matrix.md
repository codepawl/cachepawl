# Runtime Proof Sprint Scenario Matrix

Best success tier achieved: `partial_success`

This matrix pushes beyond the earlier blocker result. The strongest positive
evidence remains live stock vLLM runtime plus planner/admission replay. Direct
AVMP substitution did not run because the committed live run did not expose
Mamba state-index/state tensor contracts, and the current local environment
cannot run CUDA/NVML.

## Local Runtime Gate

| Check | Result |
| --- | --- |
| vLLM environment | `/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv` |
| vLLM version | `0.21.0` |
| Latest stable check | PyPI reported `0.21.0` as latest stable on 2026-06-05 |
| `torch.cuda.is_available()` | `false` |
| CUDA device count | `0` |
| `nvidia-smi` | failed, GPU access blocked by the operating system |

Local CUDA-required runtime scenarios are therefore environment-blocked, not
research-blocked. The GPU-machine commands in `gpu_machine_commands.sh` are the
next runnable step on a host with CUDA/NVML access.

## Config Matrix

| Scenario | max len | seqs | prompt | prefix | mamba mode | Status | Evidence |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| baseline-current-4096-seq1-short | 4096 | 1 | short | default | none | committed live run available; local rerun blocked | `research/avmp/v2/results/vllm-baseline/manifest.json` |
| local-len1024-seq1-short | 1024 | 1 | short | default | none | environment-blocked | `local-blocked-runs/current-len1024-seq1/manifest.json` |
| local-len2048-seq2-long | 2048 | 2 | long | default | none | environment-blocked | `local-blocked-runs/current-len2048-seq2/manifest.json` |
| gpu-prefix-off-mamba-none | 2048 | 1 | short and long | false | none | GPU command prepared | `gpu_machine_commands.sh` |
| gpu-prefix-on-mamba-none | 2048 | 1 | short and long | true | none | GPU command prepared | `gpu_machine_commands.sh` |
| gpu-prefix-on-mamba-align | 2048 | 1 | short and long | true | align | GPU command prepared | `gpu_machine_commands.sh` |
| gpu-prefix-on-mamba-all | 2048 | 1 | short and long | true | all | GPU command prepared | `gpu_machine_commands.sh` |
| gpu-seq4-mamba-all | 1024 | 4 | short batch | true | all | GPU command prepared | `gpu_machine_commands.sh` |

## Model Matrix

| Model | Local cache | Hybrid relevance | Status |
| --- | --- | --- | --- |
| `Zyphra/Zamba2-2.7B-instruct` | cached | hybrid Attention/Mamba, `Zamba2ForCausalLM`, `model_type=zamba2`, `mamba_d_state=64` | strongest candidate; committed live run available |
| `Qwen/Qwen2.5-1.5B-Instruct` | cached | transformer-only `Qwen2ForCausalLM`, not SSM/hybrid | rejected as positive hybrid evidence |
| `tiiuae/Falcon-H1-1.5B-Instruct` | not cached | candidate hybrid/SSM fallback | not run locally because CUDA is unavailable and download was not attempted |
| Jamba-family candidates | not cached | candidate hybrid Attention/Mamba | not run locally because CUDA is unavailable and download was not attempted |
| Hymba/RecurrentGemma/Samba candidates | not cached | candidate hybrid/SSM | not run locally because CUDA is unavailable and download was not attempted |

## Version Matrix

| Scenario | Status | Notes |
| --- | --- | --- |
| current `vllm==0.21.0` | installed and importable | Current pinned environment works at import time. |
| latest stable | same as current | PyPI reported `0.21.0` as latest stable on 2026-06-05, so no separate upgrade would change the version. |
| source/nightly | not attempted | Unsafe and low value while stable cannot reach CUDA locally. |

## Runtime Probe Matrix

| Probe | Status | Evidence |
| --- | --- | --- |
| scheduler/KV manager | observed | `vllm-runtime-contract-observation/runtime_contract_report.json` |
| worker cache tensor layout | observed | `vllm-runtime-contract-observation/runtime_contract_report.json` |
| block tables | observed | `vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` |
| request-to-block mapping | observed | `vllm-live-request-contract-observation/live_request_contract_report.json` |
| Mamba state-index | blocked in committed run; local variants environment-blocked | `vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` |
| Mamba state tensors | blocked in committed run; local variants environment-blocked | `vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` |
| state lifecycle before/during/after decode | attention lifecycle observed; Mamba lifecycle missing | `vllm-mamba-attention-contract-observation/mamba_attention_contract_report.json` |
| cache-mode/prefix state-path changes | GPU command prepared | `gpu_machine_commands.sh` |

## Strongest Positive Runtime Evidence

- Stock vLLM bounded generation smoke completed on the hybrid Zamba2 model.
- Live request admission and request-to-block mapping were observed.
- Block-pool usage changed during the request and recovered after completion.
- Planner replay measured a real reserved/useful byte gap in the observed vLLM
  cache plan.
- Attention block tables and attention metadata groups were observed.

## Substitution Result

No AVMP/default-off substitution ran. A micro substitution would require live
Mamba state-index and state tensor contracts. Those are not observable in the
committed `mamba_cache_mode=none` run, and local CUDA is blocked for `all` and
`align` cache-mode variants.

## Remaining Blocker After Exhausting Local Scenarios

The remaining blocker is mixed:

- Environment blocker for new local runtime experiments: CUDA/NVML is blocked
  by the OS.
- Runtime contract blocker for the committed live run: Mamba state-index and
  Mamba state tensors are not safely reachable through stable runtime
  attributes.

The next positive path is to run `gpu_machine_commands.sh` on the GPU host,
especially `mamba_cache_mode=all` and `mamba_cache_mode=align`, then rerun the
Mamba/attention contract probe. If Mamba state contracts become observable,
the default-off substitution probe can be designed with rollback and parity
checks.
