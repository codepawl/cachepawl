# v0.x Advisory CLI Alpha Release Checklist

Date prepared: 2026-05-28

This checklist prepares Cachepawl for an advisory CLI alpha release centered on
`cachepawl diagnose-vllm`. The release remains artifact-input only: it does not
enable runtime mutation, modify vLLM, monkeypatch vLLM internals, replace vLLM
allocators, add vLLM as a dependency, require CUDA/NVML, or claim runtime memory
savings.

## Version Recommendation

- Current package version: `0.1.0`.
- Recommended advisory alpha target: `0.2.0a1`.

Rationale: the diagnostic CLI is a new user-visible advisory surface, but the
project is still pre-alpha and runtime mutation remains out of scope. Do not
change `pyproject.toml` until the release owner confirms the target version.

## Release Scope

Included:

- `cachepawl diagnose-vllm --help`.
- `cachepawl diagnose-vllm` run from existing translated vLLM cache-plan
  observation artifacts.
- Optional `--summary-only`, `--format`, `--fail-on-waste-fraction`, and
  `--fail-on-overestimation-ratio` gates for local release checks.
- Advisory reports written as `report.json`, `summary.md`, and `manifest.json`.
- Existing 4-cell Path C advisory matrix and paper/evaluation skeleton as
  supporting evidence.

Excluded:

- vLLM runtime mutation.
- vLLM source edits, monkeypatching, or allocator replacement.
- Adding vLLM as a dependency.
- New experiments or benchmark cells.
- Empirical runtime superiority claims.

## Required Checks

Run from the repository root.

### Fresh Install Smoke

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cpu
uv run cachepawl --help
uv run python -c "import cachepawl; print(cachepawl.__version__)"
```

### Diagnostic CLI Help

```bash
uv run cachepawl diagnose-vllm --help
```

### Existing Artifact Diagnostic Run

```bash
rm -rf /tmp/cachepawl-v0x-advisory-alpha-diagnostic
uv run cachepawl diagnose-vllm \
  --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json \
  --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json \
  --output-dir /tmp/cachepawl-v0x-advisory-alpha-diagnostic
cat /tmp/cachepawl-v0x-advisory-alpha-diagnostic/report.json
cat /tmp/cachepawl-v0x-advisory-alpha-diagnostic/summary.md
cat /tmp/cachepawl-v0x-advisory-alpha-diagnostic/manifest.json
```

Optional threshold-gated run:

```bash
uv run cachepawl diagnose-vllm \
  --translated-cache-config research/avmp/v2/results/vllm-runtime-cache-plan-observation/translated_runtime_cache_config.json \
  --raw-safe-metadata research/avmp/v2/results/vllm-runtime-cache-plan-observation/raw_safe_metadata.json \
  --output-dir /tmp/cachepawl-v0x-advisory-alpha-diagnostic-gated \
  --summary-only \
  --format markdown \
  --fail-on-waste-fraction 0.5 \
  --fail-on-overestimation-ratio 2.0
```

Expected properties:

- `classification` is `planner_advisory_available`.
- `advisory_only` is `true`.
- `runtime_mutation` is `false`.
- `allocator_replacement` is `false`.
- `vllm_required` is `false` in `manifest.json`.
- The summary states that runtime savings require a future mutation hook.

### Targeted CLI Test

```bash
uv run pytest tests/cli/test_diagnose_vllm.py -q
```

### Lint

```bash
uv run ruff check .
```

### Format Check

```bash
uv run ruff format --check .
```

### Type Check

```bash
uv run mypy src/cachepawl tests research/avmp/scripts benchmarks/scripts/run_cache_probe.py benchmarks/scripts/compare_cache_planners.py benchmarks/scripts/create_planner_comparison_pack.py benchmarks/scripts/capture_vllm_baseline.py benchmarks/scripts/capture_vllm_cache_plan_observation.py benchmarks/scripts/capture_vllm_runtime_cache_plan_observation.py benchmarks/scripts/create_vllm_cache_diagnostic.py benchmarks/scripts/create_vllm_planner_dry_run_probe.py benchmarks/scripts/capture_vllm_planner_stage_observation.py benchmarks/scripts/create_vllm_planner_stage_advisory_diff.py benchmarks/scripts/create_vllm_mutation_readiness.py benchmarks/scripts/capture_vllm_runtime_contract_observation.py benchmarks/scripts/capture_vllm_live_request_contract_observation.py benchmarks/scripts/capture_vllm_mamba_attention_contract_observation.py benchmarks/scripts/run_vllm_path_c_advisory_matrix.py
```

### Full Tests

```bash
uv run pytest -q
```

### Build

```bash
uv build
```

### Diff Hygiene

```bash
git diff --check
```

## Release Decision Gates

- Package version target is confirmed by the release owner.
- `CHANGELOG.md` is updated for the selected version.
- The diagnostic artifact run is advisory-only and does not require vLLM.
- Verification commands above pass in a fresh local environment.
- Build artifacts are inspected locally but not committed unless explicitly
  requested.

## Known Limitations

- The diagnostic CLI reports planner-level advisory metrics from artifacts. It
  does not measure serving throughput, latency, model quality, or live VRAM
  savings.
- Existing Path C evidence covers a bounded 4-cell matrix for one hybrid model.
- Runtime cache substitution is deferred until missing Mamba state contracts or
  a supported vLLM integration path are available.
