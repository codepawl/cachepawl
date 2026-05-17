"""Integration tests for the AVMP paper figure generation script.

The script under test lives at ``research/avmp/scripts/generate_figures.py``.
It is on the test pythonpath via ``research/avmp`` in
``pyproject.toml`` ``[tool.pytest.ini_options].pythonpath``.

These tests are integration-flavoured: they invoke the real script on
the committed sweep aggregated.json files and assert the file set the
paper expects to consume.
"""

from __future__ import annotations

from pathlib import Path

from scripts.generate_figures import generate_all


def test_generates_four_figures_two_formats(tmp_path: Path) -> None:
    paths = generate_all(tmp_path)
    # 4 figures x 2 formats (.pdf + .png) = 8 files.
    assert len(paths) == 8
    for p in paths:
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"empty artifact: {p}"


def test_pdf_basenames_match_paper_expected(tmp_path: Path) -> None:
    paths = generate_all(tmp_path)
    pdf_basenames = {p.stem for p in paths if p.suffix == ".pdf"}
    assert pdf_basenames == {
        "fig_oom_vs_batch_size",
        "fig_oom_comparison_final",
        "fig_peak_reserved_tradeoff",
        "fig_threshold_sensitivity",
    }


def test_png_basenames_match_pdf_basenames(tmp_path: Path) -> None:
    """Every PDF has a PNG companion of the same stem for review."""

    paths = generate_all(tmp_path)
    pdf_stems = {p.stem for p in paths if p.suffix == ".pdf"}
    png_stems = {p.stem for p in paths if p.suffix == ".png"}
    assert pdf_stems == png_stems
