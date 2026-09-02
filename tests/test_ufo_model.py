"""Structural validation of the in-repo mCP UFO model (no MadGraph needed)
and the analytic LO DY formula."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def ufo():
    sys.path.insert(0, str(MODELS))
    try:
        # a fresh import even if a previous test run half-loaded it
        for m in [m for m in sys.modules if m.startswith("mcp_ufo")]:
            del sys.modules[m]
        yield importlib.import_module("mcp_ufo")
    finally:
        sys.path.remove(str(MODELS))


def test_particle_content(ufo):
    by_pdg = {p.pdg_code: p for p in ufo.all_particles}
    for pdg in (1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 21, 22, 31, -31):
        assert pdg in by_pdg, f"missing pdg {pdg}"
    chi = by_pdg[31]
    assert chi.spin == 2 and chi.color == 1
    assert chi.mass.name == "MCHI"
    # anti-quarks carry conjugate color and opposite charge
    assert by_pdg[-2].color == -3 and by_pdg[-2].charge == -by_pdg[2].charge


def test_external_parameters_have_lha_blocks(ufo):
    ext = {p.name: p for p in ufo.all_parameters if p.nature == "external"}
    assert set(ext) == {"aEWM1", "aS", "EPS", "MCHI"}
    assert ext["MCHI"].lhablock == "MASS"
    assert ext["MCHI"].lhacode == [31]
    assert ext["EPS"].lhablock == "MCPINPUTS"


def test_vertices_cover_all_charged_fermions(ufo):
    photon_vertices = [v for v in ufo.all_vertices
                       if any(p.pdg_code == 22 for p in v.particles)]
    coupled = {abs(p.pdg_code) for v in photon_vertices for p in v.particles
               if p.spin == 2}
    assert coupled == {1, 2, 3, 4, 5, 31}
    for v in photon_vertices:
        assert all(c.order == {"QED": 1} for c in v.couplings.values())


def test_chi_coupling_scales_with_eps(ufo):
    gc = next(c for c in ufo.all_couplings if c.name == "GC_a_chi")
    assert "EPS" in gc.value and "ee" in gc.value


def test_internal_parameter_expressions_evaluate(ufo):
    import cmath  # noqa: F401  (namespace for the UFO expressions)
    ns = {"cmath": cmath, "complex": complex}
    for p in ufo.all_parameters:
        if p.nature == "external":
            ns[p.name] = p.value
    for p in ufo.all_parameters:
        if p.nature == "internal":
            ns[p.name] = eval(p.value, ns)
    # e = sqrt(4 pi alpha): alpha = 1/127.9
    assert abs(ns["ee"] ** 2 / (4 * np.pi) - 1 / 127.9) < 1e-12


def test_dy_lo_partonic_formula():
    from mcpsim.physics.constants import ALPHA_EM
    from mcpsim.physics.dy_lo import GEV2_TO_PB, rpoint_check, sigma_partonic_pb

    # closed-form anchor at m = 0
    s = 100.0
    expected = 4 * np.pi * ALPHA_EM ** 2 / (3 * s) / 3.0 * GEV2_TO_PB
    assert sigma_partonic_pb(s, 0.0) == pytest.approx(expected, rel=1e-12)
    assert rpoint_check() == pytest.approx(expected, rel=1e-12)
    # threshold closes and beta suppression is monotone
    assert sigma_partonic_pb(1.0, 0.6) == 0.0
    sig = sigma_partonic_pb(np.array([10.0, 100.0]), 0.5)
    assert np.all(sig > 0) and sig[0] > sig[1] * 5   # ~1/s scaling
    # eps^2 and charge^2 scaling
    assert sigma_partonic_pb(s, 0.0, eps=1e-3) == pytest.approx(
        expected * 1e-6, rel=1e-12)
    assert sigma_partonic_pb(s, 0.0, e_q=2 / 3) == pytest.approx(
        expected * 4 / 9, rel=1e-12)
