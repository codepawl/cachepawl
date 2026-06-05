#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a GPU host where:
# - nvidia-smi exits 0
# - torch.cuda.is_available() is true in the pinned vLLM environment
# - the model cache or network access can resolve Zyphra/Zamba2-2.7B-instruct
#
# These commands are read-only vLLM observations. They do not patch vLLM, do
# not replace allocators, do not return Cachepawl plans, and do not claim AVMP
# runtime savings.

VENV_PY="${VENV_PY:-/home/nxank4/.cache/cachepawl/vllm-cachepawl-venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
OUT_ROOT="${OUT_ROOT:-research/avmp/v2/results/vllm-runtime-proof-sprint/gpu-runs}"
MODEL="${MODEL:-Zyphra/Zamba2-2.7B-instruct}"

mkdir -p "${OUT_ROOT}"

nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
"${VENV_PY}" - <<'PY'
import torch
import vllm
print({"vllm": vllm.__version__, "cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count()})
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

run_contract_probe() {
  local name="$1"
  local max_model_len="$2"
  local max_num_seqs="$3"
  local prefix_caching="$4"
  local mamba_cache_mode="$5"
  local prompt="$6"
  local max_new_tokens="$7"
  local out_dir="${OUT_ROOT}/${name}"

  mkdir -p "${out_dir}"
  "${VENV_PY}" - <<PY > "${out_dir}/raw_stdout.log" 2> "${out_dir}/raw_stderr.log"
import json
import time
import torch
from cachepawl.integrations.vllm import (
    observe_vllm_live_request_contract,
    observe_vllm_mamba_attention_contract,
    observe_vllm_runtime_cache_plan,
    observe_vllm_runtime_contracts,
)
from vllm import LLM, SamplingParams

model = ${MODEL@Q}
prompt = ${prompt@Q}
max_new_tokens = int(${max_new_tokens@Q})
params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

torch.cuda.reset_peak_memory_stats()
free_before, total_before = torch.cuda.mem_get_info()
start_load = time.perf_counter()
llm = LLM(
    model=model,
    max_model_len=int(${max_model_len@Q}),
    gpu_memory_utilization=0.7,
    max_num_seqs=int(${max_num_seqs@Q}),
    trust_remote_code=False,
    enable_prefix_caching=json.loads(${prefix_caching@Q}),
    mamba_cache_mode=${mamba_cache_mode@Q},
)
load_elapsed = time.perf_counter() - start_load
free_after_load, total_after_load = torch.cuda.mem_get_info()

runtime_plan = observe_vllm_runtime_cache_plan(llm).to_dict()
runtime_contract = observe_vllm_runtime_contracts(llm).to_dict()
live_contract = observe_vllm_live_request_contract(
    llm,
    prompt=prompt,
    sampling_params=params,
    max_new_tokens=max_new_tokens,
).to_dict()
mamba_contract = observe_vllm_mamba_attention_contract(
    llm,
    prompt=prompt,
    sampling_params=params,
    max_new_tokens=max_new_tokens,
).to_dict()

start_gen = time.perf_counter()
outputs = llm.generate([prompt], params, use_tqdm=False)
gen_elapsed = time.perf_counter() - start_gen
completion = outputs[0].outputs[0]
free_after_gen, total_after_gen = torch.cuda.mem_get_info()

payload = {
    "scenario": ${name@Q},
    "model": model,
    "max_model_len": int(${max_model_len@Q}),
    "max_num_seqs": int(${max_num_seqs@Q}),
    "enable_prefix_caching": json.loads(${prefix_caching@Q}),
    "mamba_cache_mode": ${mamba_cache_mode@Q},
    "prompt": prompt,
    "max_new_tokens": max_new_tokens,
    "load_elapsed_seconds": load_elapsed,
    "generation_elapsed_seconds": gen_elapsed,
    "generated_token_count": len(getattr(completion, "token_ids", None) or []),
    "output_text": completion.text,
    "memory": {
        "free_before": free_before,
        "total_before": total_before,
        "free_after_load": free_after_load,
        "total_after_load": total_after_load,
        "free_after_gen": free_after_gen,
        "total_after_gen": total_after_gen,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
    },
    "runtime_plan": runtime_plan,
    "runtime_contract": runtime_contract,
    "live_contract": live_contract,
    "mamba_attention_contract": mamba_contract,
    "substitution": {
        "attempted": False,
        "reason": "read-only contract matrix step"
    },
}
print("CACHEPAWL_GPU_SCENARIO_RESULT=" + json.dumps(payload, sort_keys=True))
PY

  grep 'CACHEPAWL_GPU_SCENARIO_RESULT=' "${out_dir}/raw_stdout.log" \
    | sed 's/^CACHEPAWL_GPU_SCENARIO_RESULT=//' > "${out_dir}/result.json"
}

run_contract_probe \
  "prefix-off-mamba-none-short" \
  "2048" \
  "1" \
  "false" \
  "none" \
  "Short prompt." \
  "8"

run_contract_probe \
  "prefix-on-mamba-none-long" \
  "2048" \
  "1" \
  "true" \
  "none" \
  "Cachepawl long prompt for prefix-cache and Mamba state contract observation. Repeat one bounded sentence for prefill pressure." \
  "8"

run_contract_probe \
  "prefix-on-mamba-align-long" \
  "2048" \
  "1" \
  "true" \
  "align" \
  "Cachepawl long prompt for prefix-cache and Mamba align mode contract observation. Repeat one bounded sentence for prefill pressure." \
  "8"

run_contract_probe \
  "prefix-on-mamba-all-long" \
  "2048" \
  "1" \
  "true" \
  "all" \
  "Cachepawl long prompt for prefix-cache and Mamba all mode contract observation. Repeat one bounded sentence for prefill pressure." \
  "8"

run_contract_probe \
  "seq4-mamba-all-short" \
  "1024" \
  "4" \
  "true" \
  "all" \
  "Short prompt." \
  "8"

find "${OUT_ROOT}" -name result.json -print
