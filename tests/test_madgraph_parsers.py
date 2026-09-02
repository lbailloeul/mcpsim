"""MadGraph text-processing helpers (no MG5 needed)."""

import pytest

from mcpsim.generation import madgraph
from mcpsim.generation.common import ToolchainError


def test_parse_cross_section(tmp_path):
    run = tmp_path / "Events" / "run_01"
    run.mkdir(parents=True)
    (run / "results.txt").write_text(
        "some header\n  Integrated weight (pb)  :  4.2e-03\n")
    assert madgraph.parse_cross_section(tmp_path) == pytest.approx(4.2e-3)


def test_parse_cross_section_missing(tmp_path):
    (tmp_path / "Events").mkdir()
    with pytest.raises(ToolchainError, match="could not parse cross section"):
        madgraph.parse_cross_section(tmp_path)


def test_find_lhe_prefers_latest(tmp_path):
    for run in ("run_01", "run_02"):
        d = tmp_path / "Events" / run
        d.mkdir(parents=True)
        (d / "unweighted_events.lhe.gz").write_bytes(b"")
    assert "run_02" in str(madgraph._find_lhe(tmp_path))


def test_find_lhe_missing(tmp_path):
    (tmp_path / "Events").mkdir()
    with pytest.raises(ToolchainError, match="no LHE output"):
        madgraph._find_lhe(tmp_path)
