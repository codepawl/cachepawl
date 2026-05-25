# Work Log

## 2026-05-23 — PawlKit setup for cachepawl

- Created `.pawl/` project operating folder
- Added active sprint tracker and task tracker
- Added decision log
- Added cachepawl-specific context files
- Added root `AGENTS.md` and Claude command/skill integration
- Completed Sprint 0 / T000 and opened Sprint 1 / T001 for vLLM integration
- Noted that `.agents/` and `.codex/` are read-only mounts in this workspace, so their optional adapter files could not be written here
- Verified PawlKit with `check` and `view`
- Verified repo checks: ruff, format check, mypy, pytest, and build
- No product code implemented

## 2026-05-23 — vLLM integration skeleton

- Added import-safe `cachepawl.integrations.vllm` planning skeleton
- Added frozen/slots cache-plan dataclasses and optional vLLM availability helpers
- Added focused tests under `tests/integration/vllm/`
- Updated README status wording to reflect implemented allocator prototypes and benchmark harnesses
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, and Triton deployment remain out of scope

## 2026-05-23 — baseline measurement spine

- Added `cachepawl.bench` planner-probe package with JSONL result schema
- Added CPU-safe runtime/GPU environment capture without requiring CUDA or vLLM
- Added deterministic `short-heavy`, `long-heavy`, and `mixed` synthetic workloads
- Added RTX 3060 12GB benchmark config artifact
- Added `benchmarks/scripts/run_cache_probe.py` JSONL probe CLI
- Added tests for schema validation, serialization, environment fallbacks, deterministic workloads, and CLI output
- Verified: PawlKit check, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, allocator replacement, monkeypatching, Triton kernels, and LSDR remain out of scope

## 2026-05-23 — planner comparison evidence

- Added vLLM-style padded cache-planning modeling baseline
- Added Cachepawl AVMP planner comparison path using the same synthetic workloads
- Added `benchmarks/scripts/compare_cache_planners.py` for JSONL records and Markdown/CSV summaries
- Added tests for planner behavior, schema round-trips, summary rendering, and deterministic CLI output
- Verified: PawlKit check, example comparison command, bench tests, ruff, format check, mypy, full pytest, and build
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner metric semantics correction

- Replaced ambiguous `waste_ratio` benchmark fields with explicit `overestimation_ratio` and `wasted_fraction` fields
- Defined `overestimation_ratio` as `estimated_bytes / useful_bytes`
- Defined `wasted_fraction` as `(estimated_bytes - useful_bytes) / estimated_bytes`
- Updated JSONL records, Markdown/CSV summaries, planner estimates, synthetic probe output, and tests to use the corrected names
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — planner comparison artifact pack

- Added a deterministic artifact-pack generator for the planner comparison
- Generated `benchmarks/results/rtx3060/planner-comparison/` with per-workload JSONL, combined summary, environment metadata, manifest, and reproduction README
- Added a narrow `.gitignore` allowlist for the RTX 3060 planner-comparison reference pack
- Added tests for deterministic artifact-pack generation
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — vanilla vLLM runtime baseline capture

- Added a pinned vLLM baseline capture runbook for the measurement-only runtime step
- Added `benchmarks/scripts/capture_vllm_baseline.py` to record vLLM/CUDA/GPU availability and runtime blockers without requiring vLLM
- Captured `research/avmp/v2/results/vllm-baseline/` with a structured `not_runnable` JSONL record and manifest
- Recorded that vLLM is not installed in the active uv environment, torch reports CUDA unavailable, and `nvidia-smi` cannot initialize NVML
- Runtime vLLM serving, monkeypatching, allocator replacement, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — vLLM runtime baseline infrastructure decision

- Added D004 choosing local WSL2 GPU/NVML repair before creating the pinned vLLM environment
- Updated the runtime baseline blocker artifact with the full blocker chain and selected infrastructure path
- Deferred separate Linux GPU and rented cloud GPU paths unless local WSL2 GPU access cannot be restored
- Runtime vLLM serving, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, LSDR, and real inference remain out of scope

## 2026-05-23 — WSL2 GPU visibility restored for vLLM baseline

- Updated D004 to record restored local WSL2 GPU/NVML visibility
- Re-ran the vLLM baseline capture; the artifact now records CUDA available on the NVIDIA GeForce RTX 3060 with one torch CUDA device
- Updated the remaining blocker to missing vLLM in the active Cachepawl environment
- Did not create the pinned vLLM environment or add vLLM to the main environment

## 2026-05-23 — isolated pinned vLLM environment

- Created `/tmp/vllm-cachepawl-venv` with Python 3.10
- Installed pinned `vllm==0.21.0` inside the isolated environment with `uv pip install "vllm==0.21.0" --torch-backend=auto`
- Validated vLLM import and CUDA visibility inside the pinned env on the local RTX 3060
- Re-ran `capture_vllm_baseline.py` through the pinned env with `PYTHONPATH=src`; editable Cachepawl install was not needed
- Updated `research/avmp/v2/results/vllm-baseline/` to `status=ready`
- Runtime serving, model loading, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, LSDR, and model quality evaluation remain out of scope

## 2026-05-23 — bounded vanilla vLLM model-load smoke

- Added a bounded runtime smoke mode to `benchmarks/scripts/capture_vllm_baseline.py`
- Ran the capture through `/tmp/vllm-cachepawl-venv` with `PYTHONPATH=src`; editable Cachepawl install was still not needed
- Loaded `Zyphra/Zamba2-2.7B-instruct` with vanilla `vllm==0.21.0` on the local RTX 3060 using a 1200 second timeout
- Updated `research/avmp/v2/results/vllm-baseline/` to `status=completed` for bounded model-load smoke
- Recorded vLLM observations: 5.07 GiB model memory, 2.12 GiB available KV cache memory, 11,442 GPU KV cache tokens, 2.79x max concurrency for 4,096-token requests, and 43.12% Mamba page-size padding
- Long-lived serving, generation, model-quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, and LSDR remain out of scope

## 2026-05-23 — bounded vanilla vLLM generation smoke

- Added one-prompt bounded generation capture to `benchmarks/scripts/capture_vllm_baseline.py`
- Reused `/tmp/vllm-cachepawl-venv` with pinned `vllm==0.21.0` and `PYTHONPATH=src`; editable Cachepawl install was not needed
- Captured vanilla generation for `Zyphra/Zamba2-2.7B-instruct` with `max_model_len=4096`, `gpu_memory_utilization=0.7`, `max_num_seqs=1`, and `max_new_tokens=8`
- Preserved the existing model-load smoke evidence and added a second JSONL record for bounded generation
- Recorded prompt tokens 13, generated tokens 8, elapsed 43.399474 seconds, 0.184334 tokens/sec, and 10,905,399,296 bytes available GPU memory after generation
- Long-lived serving, model-quality evaluation, monkeypatching, allocator replacement, Path C shim work, Triton kernels, copy kernels, and LSDR remain out of scope

## 2026-05-23 — Path C shim audit

- Added `research/avmp/v2/PATH_C_SHIM_AUDIT.md`
- Audited installed `vllm==0.21.0` cache planning, hybrid cache, Mamba cache, scheduler, manager, and GPU runner paths under `/tmp/vllm-cachepawl-venv`
- Identified `KVCacheSpec`, `MambaSpec`, `KVCacheConfig`, `get_kv_cache_configs`, `Scheduler.__init__`, `KVCacheManager`, `HybridKVCacheCoordinator`, `MambaManager`, and `GPUModelRunner.initialize_kv_cache` as the key integration surfaces
- Added D005 choosing observe-first translation of vLLM cache plans before any scheduler or allocator mutation
- vLLM source edits, monkeypatching, allocator replacement, Path C shim behavior, Triton kernels, copy kernels, LSDR, long-lived serving, and quality evaluation remain out of scope

## 2026-05-25 — observe-first vLLM cache-plan translator

- Added `cachepawl.integrations.vllm.translator` with import-safe, duck-typed translation for vLLM-like cache planning objects
- Added serializable Cachepawl-owned records for translated cache specs, groups, tensors, and full cache configs
- Added fake-object tests for attention specs, Mamba/state specs, hybrid cache configs, alias handling, deterministic serialization, and typed unsupported-object errors
- Updated the Path C audit limitations to record that this step observes cache plans only and performs no vLLM mutation
- vLLM source edits, monkeypatching, allocator replacement, Scheduler/KVCacheManager injection, Triton kernels, copy kernels, LSDR, serving changes, and quality evaluation remain out of scope

## 2026-05-25 — PawlKit validation path restore

- Restored the reproducible PawlKit command as `npx @codepawl/pawlkit@0.3.0`
- Documented the pinned command in `.pawl/README.md` and `.pawl/context/REPO_COMMANDS.md`
- Validated current manual `.pawl/` edits with `view` and `check`; `check` passed with 0 warnings
- Confirmed that unscoped `npx pawlkit` is the wrong package name for this repo's current tooling
