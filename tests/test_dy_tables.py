"""The LO Drell-Yan convolution and the per-material table generator, with
toy parton densities (no LHAPDF grids or `parton` needed), plus an opt-in
regeneration check against the real grids (MCPSIM_PDFSETS)."""

import os

import numpy as np
import pytest

from mcpsim.generation import dy_tables as T
from mcpsim.physics.dy_lo import sigma_hadronic_pb, sigma_partonic_pb


def toy_pdf(scale=1.0):
    """Q-independent: valence-like quarks, softer antiquarks."""
    def f(pid, x, q):
        return scale * ((1 - x) ** 3 if pid > 0 else 0.3 * (1 - x) ** 7) / x
    return f


def test_hadronic_convolution_matches_direct_quadrature():
    from scipy import integrate
    s, m, cut = 226.9, 0.8, 2.0
    beam, target = toy_pdf(), toy_pdf(0.7)
    got = sigma_hadronic_pb(m, s, beam, target, m_pair_min=cut)
    tau_min = max(4 * m * m, cut * cut) / s

    def integrand(x2, x1):
        one = np.array([x1]), np.array([x2]), np.array([np.sqrt(x1 * x2 * s)])
        lumi = sum(e2 * (beam(q, one[0], one[2]) * target(-q, one[1], one[2])
                         + beam(-q, one[0], one[2]) * target(q, one[1], one[2]))[0]
                   for q, e2 in {1: 1 / 9, 2: 4 / 9, 3: 1 / 9, 4: 4 / 9}.items())
        return lumi * sigma_partonic_pb(x1 * x2 * s, m)
    ref, _ = integrate.dblquad(integrand, tau_min, 1, lambda x1: tau_min / x1, 1,
                               epsabs=0, epsrel=1e-8)
    assert got == pytest.approx(ref, rel=1e-5)


def test_hadronic_convolution_symmetries():
    s = 752.0
    a, b = toy_pdf(), toy_pdf(0.5)
    one = sigma_hadronic_pb(1.0, s, a, b, m_pair_min=2.0)
    assert sigma_hadronic_pb(1.0, s, b, a, m_pair_min=2.0) == pytest.approx(one, rel=1e-10)
    assert sigma_hadronic_pb(1.0, s, a, b, m_pair_min=2.0, alpha=2 / 137) == pytest.approx(4 * one)
    assert sigma_hadronic_pb(14.0, s, a, b) == 0.0                 # 2 m > sqrt(s)


def test_bound_proton_weights():
    assert T.bound_proton_weights(56) == [("nCTEQ15_56_26", 1.0)]
    (lo, w_lo), (hi, w_hi) = T.bound_proton_weights(96)
    assert (lo, hi) == ("nCTEQ15_84_42", "nCTEQ15_108_54")
    assert w_hi == pytest.approx(np.log(96 / 84) / np.log(108 / 84)) and w_lo + w_hi == pytest.approx(1)
    with pytest.raises(ValueError):
        T.bound_proton_weights(250)


@pytest.fixture
def toy_grids(monkeypatch):
    """Bound-proton 'sets' with u = A and d = 1 (Q- and x-independent)."""
    def fake(name, pdfdir):
        A = float(name.split("_")[1])
        return lambda pid, x, q: np.full_like(x, {2: A, -2: 0.1, 1: 1.0, -1: 0.2}.get(pid, 0.0))
    monkeypatch.setattr(T, "_grid_pdf", fake)


def test_nucleon_pdf_isospin_and_interpolation(toy_grids):
    x = q = np.array([0.1, 0.3])
    f = T.nucleon_pdf(96, 42, "unused")
    w = np.log(96 / 84) / np.log(108 / 84)
    u_p = (1 - w) * 84 + w * 108                       # bound-proton u at A = 96
    np.testing.assert_allclose(f(2, x, q), 42 / 96 * u_p + 54 / 96 * 1.0)   # u_n = d_p
    np.testing.assert_allclose(f(1, x, q), 42 / 96 * 1.0 + 54 / 96 * u_p)   # d_n = u_p
    np.testing.assert_allclose(f(-2, x, q), 42 / 96 * 0.1 + 54 / 96 * 0.2)
    np.testing.assert_allclose(T.nucleon_pdf(1, 1, "unused")(2, x, q), 1.0)  # free proton


def test_compute_and_write_table_roundtrip(monkeypatch, tmp_path):
    from mcpsim import geometry
    from mcpsim.config import load_preset
    from mcpsim.physics.drellyan import find_table, load_table
    monkeypatch.setattr(T, "_grid_pdf", lambda name, pdfdir: toy_pdf())
    face = load_preset("darkquest").detector                  # 1 x 1 m at 40 m
    m, cols, header = T.compute_table("molybdenum", 120.0, [7.0, 0.01, 1.0, 8.0], "unused", face)
    assert m.tolist() == [0.01, 1.0, 7.0]             # sorted; 8 GeV is above sqrt(s)/2
    s, acc = cols[:, 0], cols[:, 1]
    assert s[0] > s[1] > s[2] > 0
    assert np.all((acc > 0) & (acc < 1))
    path = T.write_table(tmp_path / T.table_name("molybdenum", 120.0), m, cols, header)
    table = load_table(path)
    np.testing.assert_allclose(table.masses, m, rtol=1e-6)
    np.testing.assert_allclose(table.sigma_pb, s, rtol=1e-6)
    np.testing.assert_allclose(table.acc, acc, rtol=1e-6)
    h = table.header
    assert (h["material"], h["A"], h["Z"], h["beam_energy_gev"]) == ("molybdenum", "96", "42", "120")
    assert "nCTEQ15_84_42" in h["target_pdf"] and "nCTEQ15_1_1" in h["beam_pdf"]
    assert table.acc_detector == geometry.face_signature(face) == "1 x 1 m face at 40 m"
    assert find_table(tmp_path, 120.0, "molybdenum") == path


def test_face_acceptance_matches_the_exact_cone_and_grows_with_the_face():
    """The numerical face kernel reproduces the exact cone for a disc, and a
    1 x 1 m face takes ~1.2x the disc's acceptance (4/pi in area)."""
    from mcpsim import geometry
    from mcpsim.config import load_preset
    from mcpsim.physics.dy_lo import acceptance_per_chi
    pdf = toy_pdf()
    cfg = load_preset("darkquest")
    disc, face = cfg.legacy_detector, cfg.detector
    for m in (0.01, 3.0):
        cone = acceptance_per_chi(m, 120.0, pdf, pdf, np.arctan(0.5 / 40), m_pair_min=2.0)
        numeric = acceptance_per_chi(m, 120.0, pdf, pdf, hit_fraction=lambda r: geometry.hit_fraction(disc, r),
                                     distance_m=40.0, m_pair_min=2.0)
        square = T.detector_acceptance(m, 120.0, pdf, pdf, face)
        assert numeric == pytest.approx(cone, rel=2e-3)
        assert T.detector_acceptance(m, 120.0, pdf, pdf, disc) == pytest.approx(cone, rel=1e-12)
        assert 1.1 < square / cone < 4 / np.pi
    with pytest.raises(ValueError):
        acceptance_per_chi(1.0, 120.0, pdf, pdf)


@pytest.mark.skipif(not os.environ.get("MCPSIM_PDFSETS"),
                    reason="set MCPSIM_PDFSETS to a directory of nCTEQ15 LHAPDF grids")
@pytest.mark.parametrize("material", ["iron", "molybdenum"])
def test_shipped_tables_regenerate(material):
    pytest.importorskip("parton")
    from mcpsim import paths
    from mcpsim.physics.drellyan import find_table, load_table
    table = load_table(find_table(paths.DY_TABLES_DIR, 400.0, material))
    pick = [0, 120, 200, 240]
    detectors, _ = T.preset_dy_inputs()
    _, again, _ = T.compute_table(material, 400.0, table.masses[pick], os.environ["MCPSIM_PDFSETS"],
                                  detectors[400.0])
    np.testing.assert_allclose(again[:, 0], table.sigma_pb[pick], rtol=1e-5)
    np.testing.assert_allclose(again[:, 1], table.acc[pick], rtol=1e-5)
