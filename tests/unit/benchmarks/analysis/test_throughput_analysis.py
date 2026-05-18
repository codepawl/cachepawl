"""Tier 1 PR B: stop-rule and analysis-markdown rendering tests."""

from __future__ import annotations

from pathlib import Path

from cachepawl.benchmarks.analysis.lexicographic_rank import rank_variants
from cachepawl.benchmarks.analysis.throughput_analysis import (
    StopRuleVerdict,
    build_workload_comparisons,
    evaluate_stop_rule,
    render_analysis_markdown,
)
from cachepawl.benchmarks.compare.aggregate import AggregatedMetrics, AggregatedRow


def _row(
    *,
    variant: str,
    workload: str,
    eff_batch_p50: float = 0.0,
    goodput: float = 0.0,
    oom: float = 0.0,
    frag: float = 0.0,
) -> AggregatedRow:
    return AggregatedRow(
        variant_label=variant,
        allocator_name="x",
        workload_name=workload,
        model_spec_name="m",
        total_bytes=1024,
        n_replicates=1,
        peak_reserved_bytes_mean=0.0,
        peak_reserved_bytes_std=0.0,
        peak_reserved_bytes_min=0,
        peak_reserved_bytes_max=0,
        fragmentation_during_load_mean=frag,
        fragmentation_during_load_std=0.0,
        fragmentation_during_load_min=frag,
        fragmentation_during_load_max=frag,
        fragmentation_peak=frag,
        oom_count_mean=oom,
        oom_count_std=0.0,
        effective_batch_size_mean_median=eff_batch_p50,
        effective_batch_size_p50_median=eff_batch_p50,
        effective_batch_size_p95_median=eff_batch_p50,
        effective_batch_size_p99_median=eff_batch_p50,
        goodput_requests_per_second_median=goodput,
        completion_ratio_median=1.0,
        time_to_first_oom_seconds_median=None,
        allocate_p50_ns_median=0,
        allocate_p95_ns_median=0,
        allocate_p99_ns_median=0,
        allocator_specific_median=(),
        allocator_specific_iqr=(),
    )


def test_stop_rule_pass_batch_size() -> None:
    """Target beats baseline by >=1.05x eff_batch on 2 workloads -> PASS_BATCH_SIZE.

    The pre-registered batch-size criterion is the primary win
    condition; if it fires, the verdict prefers it even when goodput
    would also pass independently.
    """

    rows = (
        _row(variant="avmp_dynamic_b128", workload="uniform_short", eff_batch_p50=105.0),
        _row(variant="fixed_dual_mr05", workload="uniform_short", eff_batch_p50=100.0),
        _row(variant="avmp_dynamic_b128", workload="mixed_long", eff_batch_p50=110.0),
        _row(variant="fixed_dual_mr05", workload="mixed_long", eff_batch_p50=100.0),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)

    assert verdict.verdict == "PASS_BATCH_SIZE"
    assert set(verdict.batch_size_workloads_passing) == {"uniform_short", "mixed_long"}


def test_stop_rule_pass_goodput_when_batch_size_misses() -> None:
    """Batch-size criterion under threshold, but goodput on one workload >=1.10x.

    The goodput criterion is the fallback; it requires only one
    workload to clear the 1.10x bar, modeling a wallclock-throughput
    headline win.
    """

    rows = (
        _row(
            variant="avmp_dynamic_b128",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=110.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)

    assert verdict.verdict == "PASS_GOODPUT"
    assert verdict.goodput_workloads_passing == ("uniform_short",)
    assert verdict.batch_size_workloads_passing == ()


def test_stop_rule_fail_when_neither_threshold_met() -> None:
    """Target at parity on both metrics -> FAIL.

    Same eff_batch and same goodput across workloads means neither
    bucket reaches its minimum-workload count, so the throughput
    claim is not justified.
    """

    rows = (
        _row(
            variant="avmp_dynamic_b128",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
        _row(
            variant="avmp_dynamic_b128",
            workload="mixed_long",
            eff_batch_p50=102.0,
            goodput=105.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="mixed_long",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)

    assert verdict.verdict == "FAIL"
    assert verdict.batch_size_workloads_passing == ()
    assert verdict.goodput_workloads_passing == ()


def test_stop_rule_zero_baseline_treats_positive_target_as_inf() -> None:
    """A baseline with zero eff_batch and positive target yields +inf ratio.

    This matches compute_relative_improvement's convention and
    prevents division-by-zero from masking a real win.
    """

    rows = (
        _row(variant="avmp_dynamic_b128", workload="w1", eff_batch_p50=5.0, goodput=1.0),
        _row(variant="fixed_dual_mr05", workload="w1", eff_batch_p50=0.0, goodput=0.0),
        _row(variant="avmp_dynamic_b128", workload="w2", eff_batch_p50=5.0, goodput=1.0),
        _row(variant="fixed_dual_mr05", workload="w2", eff_batch_p50=0.0, goodput=0.0),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)

    # Both workloads cross 1.05x via +inf, so PASS_BATCH_SIZE fires.
    assert verdict.verdict == "PASS_BATCH_SIZE"


def test_render_analysis_emits_verdict_on_fail(tmp_path: Path) -> None:
    """A FAIL verdict still writes the markdown with the explicit verdict line."""

    rows = (
        _row(
            variant="avmp_dynamic_b128",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)
    rankings = rank_variants(aggregated)

    output = tmp_path / "throughput_analysis.md"
    render_analysis_markdown(aggregated, verdict, rankings, output)

    text = output.read_text()
    assert "**Verdict: FAIL**" in text
    assert "throughput claim is NOT justified" in text
    assert "Hypothesis evaluation" in text
    assert "lexicographic ranking" in text


def test_render_analysis_emits_verdict_on_pass(tmp_path: Path) -> None:
    """A PASS verdict writes the markdown without the FAIL escape clause."""

    rows = (
        _row(
            variant="avmp_dynamic_b128",
            workload="uniform_short",
            eff_batch_p50=120.0,
            goodput=100.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="uniform_short",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
        _row(
            variant="avmp_dynamic_b128",
            workload="mixed_long",
            eff_batch_p50=120.0,
            goodput=100.0,
        ),
        _row(
            variant="fixed_dual_mr05",
            workload="mixed_long",
            eff_batch_p50=100.0,
            goodput=100.0,
        ),
    )
    aggregated = AggregatedMetrics(rows=rows)
    verdict = evaluate_stop_rule(aggregated)
    rankings = rank_variants(aggregated)

    output = tmp_path / "throughput_analysis.md"
    render_analysis_markdown(aggregated, verdict, rankings, output)

    text = output.read_text()
    assert "**Verdict: PASS_BATCH_SIZE**" in text
    assert "throughput claim is NOT justified" not in text


def test_build_workload_comparisons_handles_missing_variants() -> None:
    """When the target variant is absent for a workload, that workload is dropped.

    Prevents the comparison loop from emitting nonsense ratios for
    workloads where only one side has data.
    """

    rows = (
        # Only baseline has w1; target absent
        _row(variant="fixed_dual_mr05", workload="w1", eff_batch_p50=100.0),
        # Both have w2
        _row(variant="avmp_dynamic_b128", workload="w2", eff_batch_p50=120.0),
        _row(variant="fixed_dual_mr05", workload="w2", eff_batch_p50=100.0),
    )
    aggregated = AggregatedMetrics(rows=rows)
    comparisons = build_workload_comparisons(
        aggregated, target="avmp_dynamic_b128", baseline="fixed_dual_mr05"
    )
    assert [c.workload_name for c in comparisons] == ["w2"]


def test_stop_rule_verdict_dataclass_is_frozen() -> None:
    """StopRuleVerdict is immutable so callers cannot mutate the conclusion."""

    verdict = StopRuleVerdict(
        target="avmp_dynamic_b128",
        baseline="fixed_dual_mr05",
        batch_size_workloads_passing=(),
        goodput_workloads_passing=(),
        verdict="FAIL",
    )
    import dataclasses

    assert dataclasses.is_dataclass(verdict)
    assert StopRuleVerdict.__dataclass_params__.frozen  # type: ignore[attr-defined]
