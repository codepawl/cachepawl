"""Tests for the workload generator."""

from __future__ import annotations

from dataclasses import replace

from cachepawl.benchmarks import (
    JAMBA_MINI_ATTN,
    JAMBA_MINI_SSM,
    PRESETS,
    AttentionLayerProfile,
    SSMLayerProfile,
    WorkloadSpec,
    generate_request_stream,
    per_sequence_ssm_bytes,
    per_token_kv_bytes,
)
from cachepawl.quant.dtypes import DType


def test_presets_expose_three_named_workloads() -> None:
    assert set(PRESETS) == {"uniform_short", "mixed_long", "agentic_burst"}


def test_preset_layer_profiles_match_jamba_mini() -> None:
    for spec in PRESETS.values():
        assert spec.attention_profile == JAMBA_MINI_ATTN
        assert spec.ssm_profile == JAMBA_MINI_SSM
        assert spec.attention_layers == 4
        assert spec.ssm_layers == 28
        assert spec.dtype is DType.BF16


def test_same_seed_produces_identical_stream() -> None:
    spec = PRESETS["uniform_short"]
    first = generate_request_stream(spec)
    second = generate_request_stream(spec)
    assert first == second


def test_different_seed_changes_stream() -> None:
    spec = PRESETS["uniform_short"]
    a = generate_request_stream(spec)
    b = generate_request_stream(replace(spec, seed=spec.seed + 1))
    assert a != b


def test_uniform_short_prompt_mean_within_tolerance() -> None:
    spec = replace(PRESETS["uniform_short"], num_requests=2000, seed=42)
    requests = generate_request_stream(spec)
    mean_prompt = sum(r.prompt_len for r in requests) / len(requests)
    expected_mean = (128 + 1024) / 2
    assert abs(mean_prompt - expected_mean) < 25.0


def test_uniform_short_request_bounds() -> None:
    spec = replace(PRESETS["uniform_short"], num_requests=500, seed=7)
    requests = generate_request_stream(spec)
    for r in requests:
        assert 128 <= r.prompt_len <= 1024
        assert 64 <= r.gen_len <= 256
        assert r.arrival_tick == r.request_id
        assert r.departure_tick == r.arrival_tick + r.prompt_len + r.gen_len


def test_mixed_long_has_heavy_tail() -> None:
    spec = replace(PRESETS["mixed_long"], num_requests=2000, seed=123)
    requests = generate_request_stream(spec)
    above_long_threshold = sum(1 for r in requests if r.prompt_len >= 16384)
    fraction_long = above_long_threshold / len(requests)
    assert 0.10 <= fraction_long <= 0.30


def test_agentic_burst_lognormal_produces_extreme_tail() -> None:
    spec = replace(PRESETS["agentic_burst"], num_requests=5000, seed=99)
    requests = generate_request_stream(spec)
    above_extreme = sum(1 for r in requests if r.prompt_len >= 10000)
    assert above_extreme >= 1
    for r in requests:
        assert 64 <= r.prompt_len <= 65536


def test_agentic_burst_arrivals_are_compressed() -> None:
    spec = replace(PRESETS["agentic_burst"], num_requests=1000, seed=11)
    requests = generate_request_stream(spec)
    span = requests[-1].arrival_tick - requests[0].arrival_tick
    one_per_tick_span = len(requests) - 1
    assert span < one_per_tick_span


def test_per_token_kv_bytes_bf16_jamba_mini_is_4096() -> None:
    bytes_per_token = per_token_kv_bytes(JAMBA_MINI_ATTN, DType.BF16)
    assert bytes_per_token == 2 * 8 * 128 * 2


def test_per_token_kv_bytes_scales_with_dtype() -> None:
    bf16 = per_token_kv_bytes(JAMBA_MINI_ATTN, DType.BF16)
    int8 = per_token_kv_bytes(JAMBA_MINI_ATTN, DType.INT8)
    fp4 = per_token_kv_bytes(JAMBA_MINI_ATTN, DType.FP4)
    assert bf16 == 2 * int8
    assert int8 == 2 * fp4


def test_per_sequence_ssm_bytes_jamba_mini() -> None:
    bf16 = per_sequence_ssm_bytes(JAMBA_MINI_SSM, DType.BF16)
    assert bf16 == 8192 * 16 * 2


def test_workload_spec_is_hashable_frozen_dataclass() -> None:
    spec = PRESETS["uniform_short"]
    assert hash(spec) == hash(spec)
    other_spec = WorkloadSpec(
        name="custom",
        num_requests=10,
        attention_layers=1,
        ssm_layers=1,
        attention_profile=AttentionLayerProfile(num_kv_heads=1, head_dim=1),
        ssm_profile=SSMLayerProfile(d_inner=1, d_state=1),
        dtype=DType.FP16,
        seed=0,
    )
    assert spec != other_spec
