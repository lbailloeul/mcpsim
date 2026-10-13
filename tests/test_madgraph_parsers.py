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
    chi-pair mass cut, and the fixed-target sqrt(s) — as massless beams of
    sqrt(s)/2 each in the centre of mass (the hidden mass_ion parameters are
    added), where MadGraph's event frame is exact."""
    from mcpsim import load_preset
    from mcpsim.generation.madgraph import MCP_PDG_DEFAULT, _patch_run_card
    card = tmp_path / "run_card.dat"
    card.write_text(" 1 = lpp1 ! beam 1 type\n 1 = lpp2 ! beam 2 type\n"
                    " 6500.0 = ebeam1 ! beam 1 total energy in GeV\n"
                    " 6500.0 = ebeam2 ! beam 2 total energy in GeV\n"
                    " nn23lo1 = pdlabel ! PDF set\n 230000 = lhaid ! if pdlabel=lhapdf\n"
                    " -1 = dynamical_scale_choice ! scale choice\n"
                    " {} = mxx_min_pdg ! min invariant mass of a pair\n"
                    " True = use_syst ! systematics\n"
                    " 10000 = nevents ! Number of unweighted events requested\n")
    original = card.read_text()
    _patch_run_card(card, load_preset("darkquest"))
    text = card.read_text()
    for line in (" 100600 = lhaid", " lhapdf = pdlabel", " 4 = dynamical_scale_choice",
                 f" {{{MCP_PDG_DEFAULT}: 2.0}} = mxx_min_pdg", " 7.532477 = ebeam1",
                 " 7.532477 = ebeam2", "  0.0 = mass_ion1\n", "  0.0 = mass_ion2\n",
                 " False = use_syst", " 10000 = nevents"):
        assert line in text, line
    card.write_text(original)
    _patch_run_card(card, load_preset("darkquest"), nevents=50000)
    assert " 50000 = nevents" in card.read_text()


def test_param_card_mass_patch_hits_only_block_mass(tmp_path):
    from mcpsim.generation.madgraph import _patch_param_card
    card = tmp_path / "param_card.dat"
    card.write_text("BLOCK MASS\n    5 4.700000e+00 # MB\n   31 1.000000e-01 # MCHI\n"
                    "Block yukawa\n   31 9.990000e+00 # not a mass\nDECAY  31 0.0 # WCHI\n")
    _patch_param_card(card, 2.5)
    text = card.read_text()
    assert "   31 2.500000e+00 # MCHI" in text          # upper-case comment kept
    assert "   31 9.990000e+00 # not a mass" in text     # other blocks untouched
    card.write_text("Block mass\n    5 4.7 # MB\n")
    with pytest.raises(ToolchainError, match="pdg 31"):
        _patch_param_card(card, 2.5)                     # no silent default-mass runs


def test_process_card_vetoes_the_z_only_if_the_model_has_one(tmp_path):
    from mcpsim import paths
    from mcpsim.generation.madgraph import PROC_CARD_TEMPLATE, _model_has_z
    assert not _model_has_z(str(paths.PROJECT_ROOT / "models" / "mcp_ufo"))   # QED only
    (tmp_path / "particles.py").write_text("Z = Particle(pdg_code = 23, name = 'Z')\n")
    assert _model_has_z(str(tmp_path))
    card = PROC_CARD_TEMPLATE.format(model="M", outdir="O", veto="")
    assert "define q = u d s c\n" in card and "define q~ = u~ d~ s~ c~\n" in card
    # both orderings: without the mirror (antiquark from the beam) sigma is
    # half the archived scans'
    assert "generate q q~ > chi chi~\nadd process q~ q > chi chi~\n" in card
    assert "output O -f\n" in card                  # reruns overwrite, no prompt


def test_set_cores_limits_madgraph_to_the_job(tmp_path):
    from mcpsim.generation.madgraph import _set_cores
    conf = tmp_path / "me5_configuration.txt"
    conf.write_text("#!  0: single machine/ 1: cluster / 2: multicore\n# run_mode = 2\n"
                    "#! Nb_core to use (None = all)\n# nb_core = None\n")
    _set_cores(conf, 4)
    text = conf.read_text()
    assert "\nrun_mode = 2\n" in text and "\nnb_core = 4\n" in text
    assert "#! Nb_core to use (None = all)" in text   # comments untouched
