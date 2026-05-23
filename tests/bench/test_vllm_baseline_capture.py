"""Tests for measurement-only vLLM baseline status capture."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from cachepawl.bench.result_schema import CacheProbeResult


def _load_capture_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_vllm_baseline",
        Path("benchmarks/scripts/capture_vllm_baseline.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_baseline_capture_writes_structured_not_runnable_result(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "vllm-baseline"
    before = "vllm" in sys.modules
    subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/capture_vllm_baseline.py",
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "1970-01-01T00:00:00Z",
            "--gpu-total-bytes",
            str(12 * 1024**3),
            "--gpu-name",
            "NVIDIA GeForce RTX 3060",
        ],
        check=True,
    )

    assert ("vllm" in sys.modules) is before
    expected_files = {"README.md", "baseline.jsonl", "manifest.json"}
    assert {path.name for path in output_dir.iterdir()} == expected_files

    result = CacheProbeResult.from_json_line((output_dir / "baseline.jsonl").read_text())
    assert result.backend == "vllm-runtime-baseline"
    assert result.workload == "runtime-baseline"
    assert result.model == "Zyphra/Zamba2-2.7B-instruct"
    assert result.estimated_bytes == 0
    assert result.reserved_bytes == 0
    assert result.useful_bytes == 0
    assert result.metadata["status"] in {"not_runnable", "ready"}
    assert "vllm_installed" in result.metadata
    assert "cuda_available" in result.metadata
    assert "blocker_chain" in result.metadata
    assert result.metadata["infrastructure_decision"] == "fix-local-wsl2-gpu-nvml-first"
    assert "python_executable" in result.metadata
    assert "pythonpath" in result.metadata
    assert result.metadata["editable_install_used"] is False
    assert result.metadata["allocator_replacement"] is False
    assert result.metadata["monkeypatching"] is False

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["artifact_name"] == "vllm-runtime-baseline"
    assert manifest["pinned_vllm_version"] == "0.21.0"
    assert manifest["model"] == "Zyphra/Zamba2-2.7B-instruct"
    assert manifest["fallback_model"] == "tiiuae/Falcon-H1-1.5B-Instruct"
    assert manifest["schema_version"] == "0.1.0"
    assert manifest["infrastructure_decision"] == "fix-local-wsl2-gpu-nvml-first"
    assert "blocker_chain" in manifest
    assert "runtime_gate" in manifest
    assert "python_executable" in manifest
    assert "pythonpath" in manifest
    assert manifest["editable_install_used"] is False
    assert "capture_vllm_baseline.py" in manifest["generation_command"]


def test_vllm_baseline_capture_is_deterministic_with_fixed_timestamp(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    base_cmd = [
        sys.executable,
        "benchmarks/scripts/capture_vllm_baseline.py",
        "--timestamp",
        "1970-01-01T00:00:00Z",
        "--gpu-total-bytes",
        str(12 * 1024**3),
        "--gpu-name",
        "NVIDIA GeForce RTX 3060",
    ]
    subprocess.run([*base_cmd, "--output-dir", str(first)], check=True)
    subprocess.run([*base_cmd, "--output-dir", str(second)], check=True)

    assert (first / "baseline.jsonl").read_text() == (second / "baseline.jsonl").read_text()


def test_vllm_baseline_capture_records_bounded_smoke_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/capture_vllm_baseline.py",
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "1970-01-01T00:00:00Z",
            "--runtime-smoke",
            "--runtime-timeout-seconds",
            "5",
            "--smoke-command",
            f"{sys.executable} -c \"print('smoke ok')\"",
        ],
        check=True,
    )

    result = CacheProbeResult.from_json_line((output_dir / "baseline.jsonl").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if result.metadata["status"] == "ready":
        assert result.metadata["runtime_smoke_status"] == "completed"
        assert result.metadata["runtime_smoke_returncode"] == 0
        assert "smoke ok" in str(result.metadata["runtime_smoke_stdout"])
        assert manifest["runtime_smoke"]["status"] == "completed"
    else:
        assert result.metadata["runtime_smoke_status"] is None
        assert manifest["runtime_smoke"] is None


def test_vllm_baseline_generation_smoke_parser_records_metrics() -> None:
    module = _load_capture_module()
    result = module._run_generation_smoke(
        model="unused",
        max_model_len=128,
        gpu_memory_utilization=0.1,
        max_num_seqs=1,
        timeout_seconds=5,
        trust_remote_code=False,
        prompt="hello",
        max_new_tokens=2,
        generation_command=(
            f"{sys.executable} -c "
            "\"print('CACHEPAWL_GENERATION_RESULT=' + "
            "__import__('json').dumps({"
            "'prompt_token_count': 3, "
            "'generated_token_count': 2, "
            "'elapsed_seconds': 0.5, "
            "'tokens_per_second': 4.0, "
            "'peak_gpu_memory_bytes': 100, "
            "'available_gpu_memory_bytes': 200"
            '}))"'
        ),
    )

    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["prompt_token_count"] == 3
    assert result["generated_token_count"] == 2
    assert result["elapsed_seconds"] == 0.5
    assert result["tokens_per_second"] == 4.0
    assert result["peak_gpu_memory_bytes"] == 100
    assert result["available_gpu_memory_bytes"] == 200
