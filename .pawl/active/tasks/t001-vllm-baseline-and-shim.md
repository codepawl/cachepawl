# T001 — Establish vLLM baseline and AVMP integration path

Project: `.pawl/active/projects/project-main.md`
Sprint: `.pawl/active/sprints/sprint-001-vllm-integration.md`
Status: Completed
Created: 2026-05-23
Updated: 2026-05-25
Completed: 2026-05-25
TTL: 30 days after completion or cancellation
Archive After: 2026-06-24
Archive Warning: 2026-06-17
Archive Reason: T001 completed; observe-first vLLM integration boundary closed

## Objective

Set up the pinned vLLM environment, capture the vanilla hybrid-cache baseline, and begin the AVMP integration path described in `research/avmp/v2/VLLM_INTEGRATION_ROADMAP.md`.

## Current Behavior

Cachepawl has Python AVMP prototypes, benchmark tooling, v2 Triton
correctness-oracle artifacts, and an import-safe
`src/cachepawl/integrations/vllm/` skeleton. Runtime vLLM allocator/shim
behavior is not implemented yet.

## Expected Behavior

The repo has reproducible vLLM baseline evidence and a clear AVMP shim implementation path, with progress recorded in `.pawl/` and research artifacts updated as results land.

## Fix Strategy

- Follow `research/avmp/v2/VLLM_DEV_SETUP.md` for the local vLLM environment.
- Pin `vllm==0.21.0`.
- Try Zamba2-2.7B-instruct first, then Falcon-H1-1.5B-Instruct if the documented swap trigger fires.
- Record vanilla baseline results before changing vLLM integration behavior.
- Implement Path C unless constructor/private-state coupling forces the Path A fork fallback.

## Progress Notes

- 2026-05-23: Added the smallest product-code step for this task: an import-safe
  `cachepawl.integrations.vllm` skeleton. It exposes typed cache-plan records and
  availability helpers without importing or depending on vLLM.
- 2026-05-23: Updated README status wording so it no longer says nothing is
  implemented.
- 2026-05-23: Added a vLLM-free baseline measurement spine under
  `cachepawl.bench`: JSONL result schema, CPU-safe environment capture,
  deterministic synthetic workloads, RTX 3060 12GB config, and a planner probe
  CLI. This supports baseline planning records before runtime vLLM serving work.
- 2026-05-23: Added first planner-comparison evidence path: a clearly labeled
  vLLM-style padded modeling baseline versus Cachepawl AVMP planning on the same
  deterministic synthetic workloads, with JSONL records and Markdown/CSV
  summaries.
- 2026-05-23: Corrected benchmark memory-efficiency metric names before
  committing planner artifacts: `overestimation_ratio` is
  `estimated_bytes / useful_bytes`, and `wasted_fraction` is
  `(estimated_bytes - useful_bytes) / estimated_bytes`.
- 2026-05-23: Generated the first reproducible planner-comparison artifact pack
  under `benchmarks/results/rtx3060/planner-comparison/` for `short-heavy`,
  `long-heavy`, and `mixed` workloads with seed 1 and deterministic planner
  runtimes.
- 2026-05-23: Started the pinned vanilla vLLM runtime baseline capture path.
  The active Cachepawl uv environment does not have vLLM installed, torch
  reports CUDA unavailable, and `nvidia-smi` cannot initialize NVML, so the
  first runtime baseline artifact is a structured `not_runnable` record.
- 2026-05-23: Accepted D004 to fix local WSL2 GPU/NVML visibility before
  creating the pinned vLLM environment or moving the runtime baseline to a
  separate GPU host. Updated the blocker artifact with the full chain:
  missing vLLM, torch CUDA unavailable, and `nvidia-smi` NVML initialization
  failure.
- 2026-05-23: Updated D004 and the runtime baseline artifact after local
  WSL2 GPU/NVML visibility was restored. `nvidia-smi` reports the RTX 3060,
  torch reports CUDA available with one device, and the remaining blocker is
  that vLLM is not installed in the active Cachepawl environment.
- 2026-05-23: Created `/tmp/vllm-cachepawl-venv`, installed pinned
  `vllm==0.21.0` only inside that isolated environment with
  `uv pip install "vllm==0.21.0" --torch-backend=auto`, validated vLLM import
  and CUDA visibility there, and re-ran the baseline capture with
  `PYTHONPATH=src` without installing Cachepawl editable into the vLLM env.
- 2026-05-23: Added and ran the bounded runtime smoke path for the primary
  hybrid target model. `Zyphra/Zamba2-2.7B-instruct` loaded successfully in
  vanilla `vllm==0.21.0` on the local RTX 3060 with no serving process,
  generation, monkeypatching, allocator replacement, or shim behavior. The
  artifact now records `status=completed` for bounded model-load smoke, with
  vLLM reporting 5.07 GiB model memory, 2.12 GiB available KV cache memory,
  11,442 GPU KV cache tokens, 2.79x max concurrency for 4,096-token requests,
  and 43.12% Mamba page-size padding.
- 2026-05-23: Added bounded generation capture to the same baseline script and
  captured one-prompt vanilla vLLM generation for
  `Zyphra/Zamba2-2.7B-instruct` with `max_new_tokens=8`, `max_num_seqs=1`,
  `max_model_len=4096`, and `gpu_memory_utilization=0.7`. The artifact keeps
  the existing model-load smoke record and adds a generation record with 13
  prompt tokens, 8 generated tokens, 43.399474 seconds elapsed, 0.184334
  tokens/sec, and 10,905,399,296 bytes available GPU memory after generation.
- 2026-05-23: Completed the first read-only Path C shim audit against the
  installed `vllm==0.21.0` package in `/tmp/vllm-cachepawl-venv`. The audit
  identified `KVCacheSpec`, `MambaSpec`, `KVCacheConfig`,
  `get_kv_cache_configs`, `Scheduler.__init__`, `KVCacheManager`,
  `HybridKVCacheCoordinator`, `MambaManager`, and
  `GPUModelRunner.initialize_kv_cache` as the key surfaces. Accepted D005 to
  observe translated vLLM cache plans before mutating scheduler or allocator
  behavior.
- 2026-05-25: Added an import-safe observe-first translator under
  `src/cachepawl/integrations/vllm/translator.py`. It accepts duck-typed
  vLLM-like `AttentionSpec`, `MambaSpec`, `KVCacheGroupSpec`,
  `KVCacheTensor`, and `KVCacheConfig` objects, emits Cachepawl-owned
  serializable snapshot records, and raises typed translation errors for
  unsupported objects without importing or mutating vLLM.
- 2026-05-25: Recreated `/tmp/vllm-cachepawl-venv`, installed pinned
  `vllm==0.21.0` there, and captured a read-only direct real-object cache-plan
  observation artifact. Reached real vLLM `AttentionSpec`, `MambaSpec`,
  `KVCacheGroupSpec`, `KVCacheTensor`, and `KVCacheConfig` dataclasses; did not
  call `get_kv_cache_configs` because it requires runtime `VllmConfig`,
  per-worker spec maps, and available-memory inputs.
- 2026-05-25: Captured and translated a runtime-resolved vanilla vLLM
  `KVCacheConfig` from
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config` using
  `/tmp/vllm-cachepawl-venv`, `vllm==0.21.0`, `PYTHONPATH=src`,
  `Zyphra/Zamba2-2.7B-instruct`, `max_model_len=4096`,
  `gpu_memory_utilization=0.7`, and `max_num_seqs=1`. The artifact records
  329 blocks, 7 cache groups, 9 cache tensors, and the same import-safe
  translator compatibility as the direct dataclass observation. No vLLM source
  edits, monkeypatching, allocator replacement, Path C mutation, or long-lived
  serving were performed.
- 2026-05-25: Converted the bounded runtime object walk into
  `observe_vllm_runtime_cache_plan(llm)` under
  `src/cachepawl/integrations/vllm/observer.py`. The helper accepts a vanilla
  vLLM `LLM` object or duck-typed equivalent, safely locates
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`,
  translates it through the existing cache-plan translator, and returns a
  deterministic serializable observation result. Unsupported runtime paths now
  return a structured `unsupported` observation instead of leaking
  `AttributeError`.
- 2026-05-25: Ran the Path C decision gate against the runtime observer
  artifact and added `research/avmp/v2/PATH_C_DECISION_GATE.md`. The translated
  runtime plan is sufficient for planner-only comparison,
  observer-in-the-loop logging, and future advisory recommendations, but not for
  replacing vLLM allocation, changing scheduler decisions, or changing tensor
  layout. Accepted D006 to implement observer-in-the-loop advisory comparison
  before any mutation.
- 2026-05-25: Compared planner-level, scheduler-construction, and worker
  allocation mutation-probe paths and added
  `research/avmp/v2/PATH_C_MUTATION_PROBE_DECISION.md`. Accepted D007 to use a
  planner-level dry-run probe next because it is the least invasive candidate
  that can test Cachepawl recommendation insertion before vLLM finalizes cache
  tensor sizes. This remains a probe-design decision only; no mutation was
  implemented.
- 2026-05-25: Closed T001 as completed for the observe-first vLLM integration
  boundary. Completed outputs include the planner benchmark spine and RTX 3060
  target-profile artifact, pinned vLLM 0.21.0 WSL2 RTX 3060 baseline, bounded
  model-load and generation smoke, real vLLM dataclass translator,
  runtime-resolved `KVCacheConfig` observer, advisory diagnostic, and
  non-mutating planner dry-run probe. Runtime mutation remains out of T001 and
  moves to later bounded probes.

## Anti-Bypass Constraints

- Do not skip the vanilla baseline and claim AVMP improvement without paired evidence.
- Do not weaken tests or reduce workloads to hide allocator regressions.
- Do not use `TritonAVMPAllocator` as the production integration path for this sprint.
- Do not silently change paper claims; update the relevant research markdown when results change.

## Done When

- [x] Vanilla vLLM bounded model-load and generation smoke are recorded for the selected model
- [x] Baseline metrics are written under `research/avmp/v2/results/`
- [x] Observe-first AVMP/vLLM path is documented with evidence; runtime mutation is deferred
- [x] Import-safe vLLM integration skeleton exists as the first shim step
- [x] Planner-only cache baseline measurement spine exists
- [x] Planner comparison emits vLLM-style padded versus Cachepawl AVMP evidence
- [x] Planner benchmark metrics use unambiguous ratio and fraction semantics
- [x] Reproducible RTX 3060 planner-comparison artifact pack exists
- [x] Pinned vanilla vLLM baseline capture path records current runtime blocker
- [x] Runtime baseline infrastructure decision is recorded
- [x] Isolated pinned vLLM environment imports vLLM and sees local CUDA device
- [x] Bounded vanilla vLLM model-load smoke is captured for the target model
- [x] Bounded vanilla vLLM one-prompt generation smoke is captured
- [x] Read-only Path C shim audit identifies vLLM 0.21.0 integration points
- [x] Observe-first cache-plan translator handles fake attention, Mamba, and hybrid vLLM-like configs
- [x] Direct real vLLM cache planning dataclasses are translated and compared against fake-object assumptions
- [x] Runtime-resolved vanilla vLLM `KVCacheConfig` is translated and compared against direct dataclass assumptions
- [x] Reusable observe-first runtime vLLM cache-plan helper handles duck-typed vanilla `LLM` objects
- [x] Path C decision gate records observer-in-the-loop advisory comparison as the next direction
- [x] Path C mutation-probe decision selects planner-level dry-run before mutation
- [x] Verification commands and skipped checks are recorded
- [x] `.pawl/logs/changelog.md` summarizes the skeleton work

## Verification

Use the commands in `.pawl/context/REPO_COMMANDS.md`. For vLLM-specific checks, record exact commands, model, GPU, and any CUDA/WSL2 limitations.

2026-05-23 skeleton verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 7 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 137 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts` — passed, 135 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 327 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

Skipped checks are CUDA-dependent tests and the deferred v2.1 copy-region kernel test.

2026-05-23 baseline measurement spine verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 19 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 146 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py` — passed, 144 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 346 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/run_cache_probe.py --workload short-heavy --backend avmp-static --seed 1 --num-requests 2 --gpu-total-bytes 12884901888` — emitted one JSONL record

2026-05-23 planner comparison verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/compare_cache_planners.py --workload mixed --seed 1 --num-requests 8 --gpu-total-bytes 12884901888` — emitted Markdown summary
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 27 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 151 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py` — passed, 149 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 354 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

2026-05-23 planner metric semantics verification:

- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
- `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 32 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/compare_cache_planners.py --workload mixed --seed 1 --num-requests 8 --gpu-total-bytes 12884901888` — emitted corrected Markdown summary with `overestimation_ratio` and `wasted_fraction`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 151 files already formatted
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py` — passed, 149 source files
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 359 passed, 12 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

2026-05-23 planner artifact pack:

- Output directory: `benchmarks/results/rtx3060/planner-comparison/`
- Files: `README.md`, `environment.json`, `manifest.json`, `summary.md`, `short-heavy.jsonl`, `long-heavy.jsonl`, `mixed.jsonl`
- Generation command: `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/create_planner_comparison_pack.py --output-dir benchmarks/results/rtx3060/planner-comparison --seed 1 --num-requests 128 --gpu-name "NVIDIA GeForce RTX 3060" --gpu-total-bytes 12884901888`
- The artifact records `vllm-style-padded` as a modeling baseline and `cachepawl-avmp` as planner-only evidence; no vLLM runtime serving or allocator replacement is included.
- `manifest.json` records artifact name, generation timestamp, fixed seed, request count, workloads, backends, target GPU profile, runtime-measurement setting, schema version, and generation command.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench -q` — 33 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 152 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py` — passed, 150 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 360 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 vanilla vLLM runtime baseline capture:

- Environment check: `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util; print('vllm_installed=' + str(importlib.util.find_spec('vllm') is not None))"` — `vllm_installed=False`
- Host check: `nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader` — failed with GPU access blocked by the operating system
- Torch check: Python 3.10.19, torch 2.12.0+cu130, CUDA unavailable, CUDA version 13.0, WSL2 platform
- Runbook: `research/avmp/v2/VLLM_BASELINE_CAPTURE.md`
- Capture command: `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --max-model-len 4096 --gpu-memory-utilization 0.9 --max-num-seqs 32 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060"`
- Output directory: `research/avmp/v2/results/vllm-baseline/`
- Files: `README.md`, `baseline.jsonl`, `manifest.json`
- Captured result: `status=not_runnable`, `reason=vllm is not installed in the active Python environment`; metadata also records CUDA unavailable and failed `nvidia-smi` NVML initialization.
- No vLLM install, runtime serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, or real inference were performed.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 2 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 362 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 vLLM runtime baseline infrastructure decision:

- Decision: `.pawl/active/decisions/d004-fix-local-wsl2-gpu-first.md`
- Selected path: fix local WSL2 GPU/NVML access first.
- Alternatives deferred: separate Linux GPU machine, rented cloud GPU, or keeping T001 runtime baseline blocked without a repair path.
- Runtime gate before pinned vLLM environment creation:
  - `nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv,noheader` must exit 0 and report the RTX 3060.
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` must report CUDA available with at least one device.
- Restored gate check:
  - `nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv,noheader` — `NVIDIA GeForce RTX 3060, 10054 MiB, 12288 MiB, 591.86`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch, importlib.util; ..."` — torch 2.12.0+cu130, CUDA build 13.0, CUDA available, one CUDA device, `NVIDIA GeForce RTX 3060`, `vllm_installed=False`
- Updated capture command: `UV_CACHE_DIR=/tmp/uv-cache uv run python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --max-model-len 4096 --gpu-memory-utilization 0.9 --max-num-seqs 32 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060"`
- Updated blocker chain: `vllm_installed=False`; GPU/NVML and torch CUDA visibility are restored.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 2 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 362 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 isolated pinned vLLM environment:

- Environment path: `/tmp/vllm-cachepawl-venv`
- Created with: `UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.10 /tmp/vllm-cachepawl-venv`
- Install command: `env UV_CACHE_DIR=/tmp/uv-cache timeout 30m uv pip install --python /tmp/vllm-cachepawl-venv/bin/python "vllm==0.21.0" --torch-backend=auto` — passed
- Installed vLLM: 0.21.0
- Pinned env validation: Python 3.10.19, vLLM 0.21.0, torch 2.11.0+cu130, CUDA build 13.0, CUDA available, one CUDA device, `NVIDIA GeForce RTX 3060`
- `nvidia-smi`: `NVIDIA GeForce RTX 3060, 10090 MiB, 12288 MiB, 591.86`
- Capture command: `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --max-model-len 4096 --gpu-memory-utilization 0.9 --max-num-seqs 32 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060"` — passed
- Cachepawl editable install into the vLLM env was not needed; `PYTHONPATH=src` was sufficient.
- Capture artifact status: `ready`; no blocker chain remains for import/CUDA readiness.
- Bounded runtime/model-load smoke was not attempted because `capture_vllm_baseline.py` does not support bounded model load or serving, and long-lived `vllm serve` is out of scope.
- No vLLM dependency was added to the main Cachepawl environment.
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util; print('main_vllm_installed=' + str(importlib.util.find_spec('vllm') is not None))"` — `main_vllm_installed=False`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 2 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 362 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
- Verification:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 2 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 362 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 bounded vLLM model-load smoke:

- Capture command: `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --max-model-len 4096 --gpu-memory-utilization 0.7 --max-num-seqs 16 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060" --runtime-smoke --runtime-timeout-seconds 1200` — passed
- Capture artifact status: `completed`; reason: `bounded vanilla vLLM model-load smoke completed`
- Runtime scope: bounded `LLM(...)` model-load smoke only; no long-lived `vllm serve`, generation, model-quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, or LSDR.
- Selected observations from vLLM logs: model loading used 5.07 GiB, available KV cache memory was 2.12 GiB, GPU KV cache size was 11,442 tokens, max concurrency for 4,096-token requests was 2.79x, and vLLM padded Mamba page size by 43.12%.
- Cachepawl editable install into the vLLM env was not needed; `PYTHONPATH=src` was sufficient.
- Verification:
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 3 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 363 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 bounded vLLM generation smoke:

- Capture command: `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_baseline.py --output-dir research/avmp/v2/results/vllm-baseline --model "Zyphra/Zamba2-2.7B-instruct" --fallback-model "tiiuae/Falcon-H1-1.5B-Instruct" --max-model-len 4096 --gpu-memory-utilization 0.7 --max-num-seqs 1 --gpu-total-bytes 12884901888 --gpu-name "NVIDIA GeForce RTX 3060" --generation-smoke --generation-timeout-seconds 1200 --generation-prompt "Cachepawl bounded vanilla vLLM baseline." --max-new-tokens 8` — passed
- Capture artifact status: `completed`; reason: `bounded vanilla vLLM generation smoke completed`
- Artifact layout: `baseline.jsonl` now contains the preserved model-load smoke record plus the bounded generation record; `manifest.json` separates `model_load_smoke` and `bounded_generation_smoke`.
- Generation metrics: prompt tokens 13, generated tokens 8, elapsed 43.399474 seconds, 0.184334 tokens/sec, available GPU memory 10,905,399,296 bytes. Peak GPU memory is recorded as unavailable because vLLM runs the engine in a child process.
- Runtime scope: one prompt and 8 generated tokens only; no long-lived `vllm serve`, quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, or LSDR.
- Verification:
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_baseline_capture.py -q` — 4 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 364 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed

2026-05-23 Path C shim audit:

- Design note: `research/avmp/v2/PATH_C_SHIM_AUDIT.md`
- Decision: `.pawl/active/decisions/d005-path-c-observe-first.md`
- Audited package root: `/tmp/vllm-cachepawl-venv/lib/python3.10/site-packages/vllm`
- Recommended next step: add import-safe translators for vLLM `KVCacheSpec`,
  `KVCacheGroupSpec`, `KVCacheTensor`, and `KVCacheConfig` before scheduler or
  allocator mutation.
- Runtime scope: read-only source audit and design only; no vLLM source edits,
  monkeypatching, allocator replacement, Path C shim behavior, Triton kernels,
  copy kernels, LSDR, long-lived serving, or quality evaluation.
- Verification:
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — passed
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 154 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 152 source files
  - `pytest` — skipped; this audit changed only docs and PawlKit records
  - `uv build` — skipped; this audit changed only docs and PawlKit records

2026-05-25 observe-first vLLM cache-plan translator:

- Added translator API:
  - `translate_kv_cache_spec(layer_name, spec)`
  - `translate_kv_cache_group(group_index, group)`
  - `translate_kv_cache_tensor(tensor)`
  - `translate_kv_cache_config(config)`
- Output records: `VllmTranslatedCacheSpec`, `VllmTranslatedCacheGroup`,
  `VllmTranslatedCacheTensor`, and `VllmTranslatedCacheConfig`, each with
  deterministic `to_dict()` output for later planner comparison.
- Supported fake-object coverage: attention-only specs, Mamba/state specs,
  hybrid cache configs with groups and tensors, layer-group name aliases, tensor
  size aliases, deterministic JSON serialization, and typed unsupported-object
  errors.
- Runtime scope: observe-first translation only; no vLLM imports, source edits,
  monkeypatching, allocator replacement, Scheduler/KVCacheManager injection,
  Triton kernels, copy kernels, LSDR, serving changes, or quality evaluation.
- Verification:
  - `pawlkit view` — skipped, `pawlkit` command not found in this shell
  - `pawlkit check` — skipped, `pawlkit` command not found in this shell
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs view` — skipped, previous fallback path no longer exists
  - `node /tmp/pawlkit-0.3.0-inspect/package/scripts/init-pawlkit.mjs check` — skipped, previous fallback path no longer exists
  - `npx pawlkit view` — skipped, npm reports no published versions for `pawlkit`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 14 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util, cachepawl.integrations.vllm as v; ..."` — `vllm_installed=False`, translator export present
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 156 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py` — passed, 154 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 371 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — passed after approved PyPI access for build requirements

2026-05-25 PawlKit validation path restore:

- Restored reproducible PawlKit validation with the pinned scoped package:
  - `npx @codepawl/pawlkit@0.3.0 view`
  - `npx @codepawl/pawlkit@0.3.0 check`
- Root cause of the temporary validation blocker: the working command was the
  scoped package `@codepawl/pawlkit@0.3.0`; `npx pawlkit` targets an unscoped
  package with no published versions, and the previous `/tmp` extraction path
  was ephemeral.
- Validation result: current manual `.pawl/` edits pass PawlKit validation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — passed
  - `npx @codepawl/pawlkit@0.3.0 check` — passed, 0 warnings

2026-05-25 vLLM cache-plan observation:

- Artifact directory:
  `research/avmp/v2/results/vllm-cache-plan-observation/`
- Files: `README.md`, `manifest.json`, `translated_cache_config.json`,
  `raw_safe_metadata.json`
- Environment setup:
  - `UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.10 /tmp/vllm-cachepawl-venv` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache timeout 30m uv pip install --python /tmp/vllm-cachepawl-venv/bin/python "vllm==0.21.0" --torch-backend=auto` — failed in sandbox due DNS
  - same install command with approved PyPI access — passed
- Capture command:
  `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_cache_plan_observation.py --output-dir research/avmp/v2/results/vllm-cache-plan-observation`
- Status: `direct_real_object_translation`
- Real objects reached: vLLM 0.21.0 `AttentionSpec`, `MambaSpec`,
  `KVCacheGroupSpec`, `KVCacheTensor`, and `KVCacheConfig`
- `get_kv_cache_configs(...)`: not called; signature requires `VllmConfig`,
  per-worker `dict[str, KVCacheSpec]` inputs, and available-memory inputs, so a
  runtime-resolved config capture remains the next observe-first step.
- Fake-vs-real comparison:
  - `AttentionSpec.page_size_bytes` matches as an observable property.
  - `AttentionSpec.dtype` is `torch.dtype`; the translator stringifies it.
  - Real `MambaSpec.shapes` and `MambaSpec.dtypes` are tuples, so the
    translator and tests were widened beyond dict-only fake assumptions.
  - Real `KVCacheGroupSpec.layer_names` and `KVCacheTensor.shared_by` are
    lists; the translator normalizes them.
  - Real `KVCacheConfig` has only `num_blocks`, `kv_cache_tensors`, and
    `kv_cache_groups`; top-level `block_size` and `cache_dtype` remain `null`.
- Runtime scope: read-only direct object translation only; no model load,
  tensors, vLLM source edits, monkeypatching, allocator replacement, scheduler
  injection, Path C mutation, Triton kernels, copy kernels, LSDR, serving
  changes, or quality evaluation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — passed
  - `npx @codepawl/pawlkit@0.3.0 check` — passed, 0 warnings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm/test_translator.py tests/bench/test_vllm_cache_plan_observation.py -q` — 9 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 158 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py` — passed, 156 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util; ..."` — `main_vllm_installed=False`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 373 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — failed in sandbox due DNS for `hatchling>=1.25`; passed after approved PyPI access

2026-05-25 runtime vLLM cache-plan observation:

- Artifact directory:
  `research/avmp/v2/results/vllm-runtime-cache-plan-observation/`
- Files: `README.md`, `manifest.json`, `translated_runtime_cache_config.json`,
  `raw_safe_metadata.json`
- Capture command:
  `PYTHONPATH=src /tmp/vllm-cachepawl-venv/bin/python benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py --output-dir research/avmp/v2/results/vllm-runtime-cache-plan-observation --model Zyphra/Zamba2-2.7B-instruct --max-model-len 4096 --gpu-memory-utilization 0.7 --max-num-seqs 1 --timeout-seconds 1200`
- Status: `runtime_resolved_translation`
- Runtime path reached:
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`
- Runtime output: 329 blocks, 7 cache groups, 9 cache tensors, 63 layers,
  6,881,280 total page-size bytes, and 5,104,128 total useful bytes.
- Runtime-vs-direct comparison:
  - Direct observation proved the translator handles real vLLM dataclasses.
  - Runtime observation proved the same translator handles the post-planning
    `Scheduler.kv_cache_config` object after vanilla offline `LLM`
    initialization.
  - Runtime planning resolves `num_blocks`, cache group count, tensor count,
    and available GPU memory values that direct dataclass construction could
    not provide.
- Runtime scope: read-only runtime observation only; no vLLM source edits,
  monkeypatching, allocator replacement, scheduler injection, Path C mutation,
  long-lived serving, Triton kernels, copy kernels, LSDR, serving changes, or
  quality evaluation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — passed after approved registry access, 0 warnings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_runtime_cache_plan_observation.py tests/bench/test_vllm_cache_plan_observation.py tests/integration/vllm/test_translator.py -q` — 10 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 160 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py` — passed, 158 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util; ..."` — `main_vllm_installed=False`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 374 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — failed in sandbox due DNS for `hatchling>=1.25`; passed after approved PyPI access

2026-05-25 reusable runtime vLLM cache-plan observer:

- Added API:
  - `observe_vllm_runtime_cache_plan(llm)`
  - `VllmRuntimeCacheObservation`
  - `RUNTIME_KV_CACHE_CONFIG_PATH`
- Supported runtime path:
  `LLM.llm_engine.engine_core.engine_core.scheduler.kv_cache_config`
- Unsupported-path behavior: missing `llm_engine`, nested `engine_core`,
  `scheduler`, or `kv_cache_config` returns a structured `unsupported`
  observation with `runtime_resolved_kv_cache_config=false`.
- Artifact compatibility: no committed runtime artifact files changed; the
  capture script still writes `manifest.json`,
  `translated_runtime_cache_config.json`, `raw_safe_metadata.json`, and
  `README.md` with the existing schema.
- Runtime scope: observe-only helper and script refactor; no vLLM source edits,
  monkeypatching, allocator replacement, scheduler injection, Path C mutation,
  long-lived serving, Triton kernels, copy kernels, LSDR, serving changes, or
  quality evaluation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — failed in sandbox due npm DNS; passed after approved registry access, 0 warnings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 18 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_runtime_cache_plan_observation.py -q` — 1 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 162 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py` — passed, 160 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util, cachepawl.integrations.vllm as v; ..."` — `main_vllm_installed=False`, `observer_export=True`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 377 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — failed in sandbox due DNS for `hatchling>=1.25`; passed after approved PyPI access

2026-05-25 Path C decision gate:

- Added `research/avmp/v2/PATH_C_DECISION_GATE.md`.
- Added D006:
  `.pawl/active/decisions/d006-path-c-observer-in-loop-first.md`.
- Decision: proceed with observer-in-the-loop advisory comparison before any
  scheduler, manager, allocator, or worker allocation mutation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — failed in sandbox due npm DNS; passed after approved registry access, 0 warnings
  - `ruff`, `ruff format`, `mypy`, `pytest`, and `uv build` — skipped because this step changed only Markdown and PawlKit records

2026-05-25 vLLM runtime cache diagnostic:

- Added API:
  - `advise_vllm_runtime_cache_plan(translated_cache_config, raw_safe_metadata=...)`
  - `VllmCacheAdvisoryReport`
  - `VllmCacheGroupAdvisory`
- Added artifact generator:
  `benchmarks/scripts/create_vllm_cache_diagnostic.py`.
- Generated artifact directory:
  `research/avmp/v2/results/vllm-runtime-cache-diagnostic/`.
- Files: `manifest.json`, `report.json`, `summary.md`.
- Classification: `planner_advisory_available`, with `observe_only` and
  `mutation_required_for_runtime_effect` flags.
- Key metrics:
  - `num_blocks`: 329
  - `cache_group_count`: 7
  - `cache_tensor_count`: 9
  - `layer_count`: 63
  - `available_kv_cache_gpu_memory_bytes`: 2,915,421,184
  - `observed_reserved_bytes`: 2,910,781,440
  - `observed_useful_bytes`: 1,679,258,112
  - `cachepawl_recommended_bytes`: 1,679,258,112
  - `advisory_savings_bytes`: 1,231,523,328
  - `overestimation_ratio`: 1.7333734577189286
  - `wasted_fraction`: 0.4230902777777778
- Missing fields for mutation: stable scheduler/planner construction hook,
  allocator or `KVCacheManager` replacement control point, worker tensor
  allocation layout control point, runtime request-to-block assignment control,
  and Mamba state-index plus attention view rewrite contract.
- Runtime scope: advisory diagnostics only; no vLLM source edits,
  monkeypatching, allocator replacement, scheduler/manager/worker mutation,
  Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR,
  serving changes, or quality evaluation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — failed in sandbox due npm DNS; passed after approved registry access, 0 warnings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 21 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_cache_diagnostic.py -q` — 1 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 166 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py` — passed, 164 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util, cachepawl.integrations.vllm as v; ..."` — `main_vllm_installed=False`, `advisory_export=True`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 381 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — failed in sandbox due DNS for `hatchling>=1.25`; passed after approved PyPI access

2026-05-25 Path C mutation-probe decision:

- Added `research/avmp/v2/PATH_C_MUTATION_PROBE_DECISION.md`.
- Added D007:
  `.pawl/active/decisions/d007-use-planner-level-dry-run-probe.md`.
- Compared:
  - planner-level hook: earliest candidate to test insertion before finalized
    `KVCacheConfig` tensor sizes; medium correctness risk, medium-high private
    API risk, good RTX 3060 dry-run testability, small dry-run footprint.
  - scheduler construction hook: can affect manager behavior but likely too
    late for finalized padded tensor sizes; high correctness and private API
    risk because vLLM directly constructs `KVCacheManager`.
  - worker allocation hook: can affect tensor layout but has very high
    correctness risk around attention views, Mamba state indices, block tables,
    and backend metadata.
- Decision: use a planner-level dry-run probe next. It should observe planner
  inputs/outputs, compute a Cachepawl proposed alternate plan, and persist both
  without returning the alternate plan to vLLM.
- Runtime scope: decision/probe design only; no vLLM source edits,
  monkeypatching, allocator replacement, scheduler/manager/worker mutation,
  Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR,
  serving changes, or quality evaluation.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — failed in sandbox due npm DNS; passed after approved registry access, 0 warnings
  - `ruff`, `ruff format`, `mypy`, `pytest`, and `uv build` — skipped because this step changed only Markdown and PawlKit records

2026-05-25 vLLM planner dry-run probe:

- Added API:
  - `dry_run_vllm_planner_probe(translated_or_observation, raw_safe_metadata=...)`
  - `VllmPlannerDryRunResult`
  - `VllmDryRunGroupProposal`
- Added artifact generator:
  `benchmarks/scripts/create_vllm_planner_dry_run_probe.py`.
- Generated artifact directory:
  `research/avmp/v2/results/vllm-planner-dry-run-probe/`.
- Files: `README.md`, `manifest.json`, `dry_run_result.json`, `summary.md`.
- Key metrics:
  - `status`: `planner_dry_run_available`
  - `safe_for_advisory_only`: `true`
  - `returned_to_vllm`: `false`
  - `vllm_behavior_changed`: `false`
  - `vanilla_observed_reserved_bytes`: 2,910,781,440
  - `vanilla_observed_useful_bytes`: 1,679,258,112
  - `cachepawl_proposed_reserved_bytes`: 1,679,258,112
  - `estimated_savings_bytes`: 1,231,523,328
  - `overestimation_ratio`: 1.7333734577189286
  - `wasted_fraction`: 0.4230902777777778
- Missing fields for mutation: stable scheduler/planner construction hook,
  allocator or `KVCacheManager` replacement control point, worker tensor layout
  control point, runtime request-to-block assignment control, and Mamba
  state-index plus attention view rewrite contract.
- Runtime scope: planner dry-run artifact only; no vLLM source edits,
  monkeypatching, allocator replacement, scheduler/manager/worker mutation,
  Path C mutation, long-lived serving, Triton kernels, copy kernels, LSDR,
  serving changes, quality evaluation, or model inference rerun.
- Verification:
  - `npx @codepawl/pawlkit@0.3.0 view` — failed in sandbox due npm DNS; passed after approved registry access
  - `npx @codepawl/pawlkit@0.3.0 check` — failed in sandbox due npm DNS; passed after approved registry access, 0 warnings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration/vllm -q` — 24 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bench/test_vllm_planner_dry_run_probe.py -q` — 1 passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .` — 170 files already formatted
  - `UV_CACHE_DIR=/tmp/uv-cache uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py` — passed, 168 source files
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util, cachepawl.integrations.vllm as v; ..."` — `main_vllm_installed=False`, `dry_run_export=True`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — 385 passed, 12 skipped
  - `UV_CACHE_DIR=/tmp/uv-cache uv build` — failed in sandbox due DNS for `hatchling>=1.25`; passed after approved PyPI access

## Regression Coverage

Added focused tests under `tests/integration/vllm/` for import safety, optional vLLM
availability probing, frozen/slots dataclass behavior, and Jamba reference cache-plan
translation. Runtime vLLM import/subclass tests remain pending until the vLLM venv is
available.

Added focused tests under `tests/bench/` for cache-probe schema round-trips and
validation, CPU-safe environment capture, deterministic synthetic workload output,
and CLI JSONL emission.

Added planner-comparison tests under `tests/bench/` for vLLM-style padded baseline
behavior, Cachepawl AVMP comparison behavior, schema round-trips, Markdown/CSV
summary rendering, and deterministic CLI JSONL output.

Added metric-semantics assertions for `overestimation_ratio` and
`wasted_fraction`, including the mixed workload example where the vLLM-style
padded baseline reports about 3.1266x overestimation and about 0.6802 wasted
fraction.

## Next Suggested Task

T002 — Run AVMP-enabled vLLM comparison once the baseline and shim are in place.
