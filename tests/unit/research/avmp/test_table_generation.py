"""Integration tests for the AVMP paper table generation script.

The script under test lives at ``research/avmp/scripts/generate_tables.py``.
"""

from __future__ import annotations

from pathlib import Path

from scripts.generate_tables import generate_all

_EXPECTED_TABLE_FILES: frozenset[str] = frozenset(
    {
        "table_baseline_comparison.tex",
        "table_per_workload_winner.tex",
        "table_parameter_defaults.tex",
        "table_stage1_batchsize.tex",
        "table_stage2_threshold.tex",
    }
)


def test_generates_five_tables_with_expected_filenames(tmp_path: Path) -> None:
    paths = generate_all(tmp_path)
    assert len(paths) == 5
    assert {p.name for p in paths} == _EXPECTED_TABLE_FILES


def test_every_table_has_booktabs_skeleton(tmp_path: Path) -> None:
    paths = generate_all(tmp_path)
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in text, f"{p.name} missing \\begin{{tabular}}"
        assert "\\end{tabular}" in text, f"{p.name} missing \\end{{tabular}}"
        assert "\\toprule" in text, f"{p.name} missing \\toprule"
        assert "\\midrule" in text, f"{p.name} missing \\midrule"
        assert "\\bottomrule" in text, f"{p.name} missing \\bottomrule"


def test_tables_contain_no_em_dashes(tmp_path: Path) -> None:
    """No em dashes (U+2014). LaTeX dashes must use ``--`` instead."""

    paths = generate_all(tmp_path)
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert "—" not in text, f"{p.name} contains an em dash"


def test_underscores_in_identifiers_are_escaped(tmp_path: Path) -> None:
    """Identifiers like ``fixed_dual_mr05`` must escape underscores for LaTeX."""

    paths = generate_all(tmp_path)
    baseline_path = next(p for p in paths if p.name == "table_baseline_comparison.tex")
    text = baseline_path.read_text(encoding="utf-8")
    # The escaped form must be present; the raw form must not.
    assert "fixed\\_dual\\_mr05" in text
    # A raw underscore could legitimately appear inside a LaTeX command,
    # so only check that the bare variant label does not.
    assert "fixed_dual_mr05" not in text
