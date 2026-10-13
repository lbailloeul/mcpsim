"""Tests for the parts of the generation layer that need no heavy toolchain:
the LHE acceptance parser and the bremsstrahlung global-patcher (text only)."""

import numpy as np
import pytest

from mcpsim import load_preset, paths
from mcpsim.generation import brem_run, mesons
from mcpsim.generation.madgraph import MCP_PDG_DEFAULT, parse_lhe_acceptance

_SOURCE_REPO_PRESENT = paths.source_repo().exists()

# Two-event LHE: a forward pair (both chi on the face), a wide-angle pair (none).
_LHE = f"""\
<LesHouchesEvents version="3.0">
<init>
</init>
<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 6.0e+01 6.0e+01 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+01 1.0e+01 0 0 -1
 {MCP_PDG_DEFAULT} 1 1 2 0 0 0.0 0.0 5.0e+01 5.0e+01 1.0e-01 0 1
-{MCP_PDG_DEFAULT} 1 1 2 0 0 0.0 0.0 1.0e+00 1.0e+00 1.0e-01 0 -1
</event>
<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 6.0e+01 6.0e+01 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+01 1.0e+01 0 0 -1
 {MCP_PDG_DEFAULT} 1 1 2 0 0 5.0e+01 0.0 1.0e+00 5.0e+01 1.0e-01 0 1
-{MCP_PDG_DEFAULT} 1 1 2 0 0 -5.0e+01 0.0 1.0e+00 5.0e+01 1.0e-01 0 -1
</event>
</LesHouchesEvents>
"""


def test_parse_lhe_acceptance(tmp_path):
    lhe = tmp_path / "events.lhe"
    lhe.write_text(_LHE)
    cfg = load_preset("darkquest")  # 1 x 1 m face at 40 m
    n_total, n_hits = parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg)
    assert n_total == 2
    assert n_hits == 2  # both chi of the forward pair; per-chi acceptance 2 / (2 * 2)


@pytest.mark.skipif(not _SOURCE_REPO_PRESENT, reason="source repo (mCP_brem.py) unavailable")
def test_brem_patcher_rewrites_globals(tmp_path):
    cfg = load_preset("ship")  # 400 GeV, molybdenum -> different globals than the script default
    patched = brem_run._prepare_patched_script(cfg, tmp_path)
    text = patched.read_text()
    # Patched copy lives in the work dir, not the source repo.
    assert patched.parent == tmp_path
    assert f"{cfg.beam.n_pot:.6e}" in text          # N_POT updated (2.000000e+20)
    assert "95.95" in text                           # molybdenum A
    # sqrt(s) recomputed for 400 GeV, no longer the 15.06 GeV default.
    assert "15.06)**2" not in text and "15.06**2" not in text
    # The output header names the nucleus sigma_pA used, so the readers do not
    # rescale the regenerated grids by the hard-coded iron A.
    assert '"A_target = 95.95 g/mol' in text
    assert '"A_Fe = 55.845 g/mol' not in text


def test_meson_backend_dispatch():
    assert mesons.choose_backend(load_preset("darkquest")) == "pythia"
    assert mesons.choose_backend(load_preset("ship")) == "pythia"
    assert mesons.choose_backend(load_preset("lanl_12bar")) == "burman_smith"
