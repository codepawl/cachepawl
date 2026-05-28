# v0.x Advisory CLI Alpha Release Checklist

Date prepared: 2026-05-28

This checklist prepares Cachepawl for an advisory CLI alpha release centered on
`cachepawl diagnose-vllm`. The release remains artifact-input only: it does not
enable runtime mutation, modify vLLM, monkeypatch vLLM internals, replace vLLM
allocators, add vLLM as a dependency, require CUDA/NVML, or claim runtime memory
savings.

## Version

- Package version: `0.2.0a1`.

Rationale: the diagnostic CLI is a new user-visible advisory surface, but the
project is still pre-alpha and runtime mutation remains out of scope. The
release owner confirmed `0.2.0a1` as the advisory CLI alpha target.

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

Expected artifacts for `0.2.0a1`:

- `dist/cachepawl-0.2.0a1.tar.gz`
- `dist/cachepawl-0.2.0a1-py3-none-any.whl`

### Diff Hygiene

```bash
git diff --check
```

## GitHub Actions Release Automation

Cachepawl uses GitHub Actions for long-term CI/CD.

### Continuous Integration

The `.github/workflows/ci.yml` workflow runs on pushes to `main` and pull
requests. It uses `uv`, Python 3.10 and 3.12, the CPU PyTorch wheel index, and
the same advisory release checks listed above. CUDA-dependent tests are allowed
to skip naturally on GitHub-hosted runners.

### PyPI Trusted Publishing Setup

Configure PyPI Trusted Publishing before pushing a release tag:

- PyPI project: `cachepawl`.
- Owner: `codepawl`.
- Repository: `cachepawl`.
- Workflow name: `publish.yml`.
- Environment name: `pypi`.

Do not add a PyPI API token or password to GitHub Secrets. The publish workflow
uses GitHub OIDC with `id-token: write` and
`pypa/gh-action-pypi-publish`.

### Tag-Based Publish Flow

After local release checks pass and the version bump commit is on `main`, create
and push the release tag:

```bash
git tag -a v0.2.0a1 -m "Cachepawl advisory CLI alpha v0.2.0a1"
git push origin main
git push origin v0.2.0a1
```

The `.github/workflows/publish.yml` workflow runs only for tags matching `v*`.
It verifies that the tag version matches `pyproject.toml`; for example,
`v0.2.0a1` must match package version `0.2.0a1`. If the tag and package
version differ, the workflow exits before building or publishing.

After pushing the tag:

- Verify the GitHub Actions publish workflow completed successfully.
- Verify the PyPI release page contains the expected sdist and wheel.
- Verify install from PyPI in a fresh environment:

```bash
uv venv /tmp/cachepawl-pypi-smoke
UV_PROJECT_ENVIRONMENT=/tmp/cachepawl-pypi-smoke uv pip install cachepawl==0.2.0a1
UV_PROJECT_ENVIRONMENT=/tmp/cachepawl-pypi-smoke uv run python -c "import cachepawl; print(cachepawl.__version__)"
UV_PROJECT_ENVIRONMENT=/tmp/cachepawl-pypi-smoke uv run cachepawl diagnose-vllm --help
```

Manual `twine upload` is a fallback only for a Trusted Publishing outage or
explicit release-owner decision. Prefer the tag-based GitHub Actions workflow.

## Release Decision Gates

- Package version target is confirmed by the release owner: `0.2.0a1`.
- `CHANGELOG.md` is updated for `0.2.0a1`.
- The diagnostic artifact run is advisory-only and does not require vLLM.
- Verification commands above pass in a fresh local environment.
- PyPI Trusted Publishing is configured for the `pypi` GitHub environment.
- Build artifacts are inspected locally but not committed unless explicitly
  requested.

## Known Limitations

- The diagnostic CLI reports planner-level advisory metrics from artifacts. It
  does not measure serving throughput, latency, model quality, or live VRAM
  savings.
- Existing Path C evidence covers a bounded 4-cell matrix for one hybrid model.
- Runtime cache substitution is deferred until missing Mamba state contracts or
  a supported vLLM integration path are available.
