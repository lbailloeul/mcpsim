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


def test_run_card_patch_reproduces_archived_settings(tmp_path):
    """The DY run card gets the settings that reproduce the archived scans:
    nCTEQ15_56_26 (lhaid 100600) for both beams, mu = sqrt(s_hat), a 2 GeV
    chi-pair mass cut, the beam energy, and a target at rest."""
    from mcpsim import load_preset
    from mcpsim.generation.madgraph import MCP_PDG_DEFAULT, _patch_run_card
    card = tmp_path / "run_card.dat"
    card.write_text(" 1 = lpp1 ! beam 1 type\n 1 = lpp2 ! beam 2 type\n"
                    " 6500.0 = ebeam1 ! beam 1 total energy in GeV\n"
                    " 6500.0 = ebeam2 ! beam 2 total energy in GeV\n"
                    " nn23lo1 = pdlabel ! PDF set\n 230000 = lhaid ! if pdlabel=lhapdf\n"
                    " -1 = dynamical_scale_choice ! scale choice\n"
                    " {} = mxx_min_pdg ! min invariant mass of a pair\n"
                    " True = use_syst ! systematics\n")
    _patch_run_card(card, load_preset("darkquest"))
    text = card.read_text()
    for line in (" 100600 = lhaid", " lhapdf = pdlabel", " 4 = dynamical_scale_choice",
                 f" {{{MCP_PDG_DEFAULT}: 2.0}} = mxx_min_pdg", " 120.0 = ebeam1", " 0.938 = ebeam2",
                 " False = use_syst"):
        assert line in text, line
