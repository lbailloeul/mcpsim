"""Analytic LO Drell-Yan partonic cross section (the MadGraph-free tier of
the DY validation).

    sigma(q qbar -> gamma* -> chi chibar)
      = (4 pi alpha^2 / 3 s_hat) * Q_q^2 * eps^2 * (1/N_c)
        * beta * (1 + 2 m_chi^2 / s_hat) / normalization of massless leptons,
    beta = sqrt(1 - 4 m_chi^2 / s_hat)

The 1/N_c = 1/3 averages the initial quark colors. At eps = 1, m_chi -> 0 and
Q_q = 1 this reduces to the textbook sigma(e+e- -> mu+ mu-) = 4 pi alpha^2 /
(3 s) divided by 3 — the standard qqbar -> l+l- result. The partonic formula
alone anchors the UFO-model muon-pair benchmark (see models/mcp_ufo/__init__.py);
sigma_hadronic_pb convolves it with parton densities. With the archived scans'
settings (iron bound-proton nPDF for both beams, mu_F = M, M >= 2 GeV, alpha =
1/127.9) it reproduces the MadGraph cross sections to 0.1% (DarkQuest) and
0.2-0.7% (SHiP), and it builds mcpsim's per-material tables
(generation/dy_tables.py).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .constants import ALPHA_EM, M_PROTON

GEV2_TO_PB = 3.893793721e8   # (hbar c)^2 in GeV^2 * pb
N_C = 3.0


# Squared charges of the flavours in MadGraph's `define q = u d s c` (PDG ids).
QUARK_CHARGE2 = {1: 1.0 / 9.0, 2: 4.0 / 9.0, 3: 1.0 / 9.0, 4: 4.0 / 9.0}

# A parton density f(pid, x, Q) -> f(x, Q): the number density (not x*f),
# vectorised over equal-length x and Q arrays.
PDF = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


def sigma_partonic_pb(s_hat_gev2, m_chi: float, eps: float = 1.0,
                      e_q: float = 1.0, alpha: float = ALPHA_EM) -> np.ndarray:
    """LO q qbar -> gamma* -> chi chibar cross section in pb.

    s_hat_gev2: partonic Mandelstam s [GeV^2] (scalar or array);
    e_q: quark charge in units of e (2/3 or -1/3).
    Color-averaged over the initial q qbar (x 1/3).
    """
    s = np.asarray(s_hat_gev2, dtype=float)
    x = np.zeros_like(s)
    open_ = s > 4.0 * m_chi ** 2
    ss = s[open_]
    beta = np.sqrt(1.0 - 4.0 * m_chi ** 2 / ss)
    x[open_] = (4.0 * np.pi * alpha ** 2 / (3.0 * ss)
                * e_q ** 2 * eps ** 2 / N_C
                * beta * (1.0 + 2.0 * m_chi ** 2 / ss))
    return x * GEV2_TO_PB


def fixed_target_s(energy_gev: float) -> float:
    """s for a proton beam on a nucleon at rest: m_p^2 + m_N^2 + 2 E m_N."""
    return 2.0 * M_PROTON ** 2 + 2.0 * energy_gev * M_PROTON


def fixed_target_y_cm(energy_gev: float) -> float:
    """Rapidity of the beam-nucleon centre of mass in the target rest frame."""
    e = energy_gev + M_PROTON
    p = np.sqrt(energy_gev ** 2 - M_PROTON ** 2)
    return 0.5 * float(np.log((e + p) / (e - p)))


def _nodes(m_chi: float, s_gev2: float, m_pair_min: float, n_tau: int, n_y: int):
    """Gauss-Legendre nodes in ln(tau) over pair masses M >= max(2 m_chi,
    m_pair_min) and in the pair rapidity y (|y| < -ln(tau)/2), with the
    measure dtau dy = dx1 dx2. Returns (tau, y, x1, x2, Q, weight), the last
    five flattened over (tau, y); None above the kinematic limit."""
    tau_min = max(4.0 * m_chi ** 2, m_pair_min ** 2) / s_gev2
    if tau_min >= 1.0:
        return None
    t_n, t_w = np.polynomial.legendre.leggauss(n_tau)
    y_n, y_w = np.polynomial.legendre.leggauss(n_y)
    half = -0.5 * np.log(tau_min)                       # ln(tau) in [-2 half, 0]
    tau = np.exp(half * (t_n - 1.0))
    y_max = -0.5 * np.log(tau)
    y = (y_max[:, None] * y_n[None, :]).ravel()
    tau_f = np.repeat(tau, n_y)
    weight = ((half * t_w * tau * y_max)[:, None] * y_w[None, :]).ravel()
    x1, x2 = np.sqrt(tau_f) * np.exp(y), np.sqrt(tau_f) * np.exp(-y)
    return tau_f, y, x1, x2, np.sqrt(tau_f * s_gev2), weight


def _luminosity(beam_pdf: PDF, target_pdf: PDF, x1, x2, q) -> np.ndarray:
    lumi = np.zeros_like(x1)
    for pid, e2 in QUARK_CHARGE2.items():
        lumi += e2 * (beam_pdf(pid, x1, q) * target_pdf(-pid, x2, q)
                      + beam_pdf(-pid, x1, q) * target_pdf(pid, x2, q))
    return lumi


def _differential_pb(m_chi, s_gev2, beam_pdf, target_pdf, m_pair_min, alpha, n_tau, n_y):
    """(tau, y, dsigma) on the integration nodes; dsigma sums to sigma in pb."""
    nodes = _nodes(m_chi, s_gev2, m_pair_min, n_tau, n_y)
    if nodes is None:
        return None
    tau, y, x1, x2, q, weight = nodes
    dsigma = (weight * _luminosity(beam_pdf, target_pdf, x1, x2, q)
              * sigma_partonic_pb(tau * s_gev2, m_chi, alpha=alpha))
    return tau, y, dsigma


def sigma_hadronic_pb(m_chi: float, s_gev2: float, beam_pdf: PDF, target_pdf: PDF,
                      *, m_pair_min: float = 0.0, alpha: float = ALPHA_EM,
                      n_tau: int = 96, n_y: int = 64) -> float:
    """LO sigma(beam + target nucleon -> chi chibar + X) in pb at eps = 1.

        sigma = int dtau dy sum_q [f_q(x1) fbar_q(x2) + fbar_q(x1) f_q(x2)]
                                  * sigma_hat_q(tau s),     x1,2 = sqrt(tau) e^(+-y),

    over pair masses M = sqrt(tau s) >= max(2 m_chi, m_pair_min), with the
    factorisation scale at the pair mass (mu_F = M, MadGraph's
    dynamical_scale_choice 4). The PDFs are per nucleon, so for a nuclear
    target this is the cross section per target nucleon. Gauss-Legendre in
    ln(tau) and in the pair rapidity y.
    """
    d = _differential_pb(m_chi, s_gev2, beam_pdf, target_pdf, m_pair_min, alpha, n_tau, n_y)
    return 0.0 if d is None else float(d[2].sum())


def cone_fraction(beta, rapidity, theta_max: float):
    """Share of chi decays (dN/dcos ~ 1 + cos^2 + (1 - beta^2) sin^2 about the
    beam axis, the gamma* -> fermion pair distribution) whose lab direction is
    within theta_max of the beam, for a pair at zero pT and lab rapidity
    `rapidity`. Exact: the accepted cos(theta*) range solves a quadratic.
    Vectorised over beta and rapidity."""
    beta, rapidity = np.broadcast_arrays(np.asarray(beta, float), np.asarray(rapidity, float))
    s_y, c_y, t2 = np.sinh(rapidity), np.cosh(rapidity), np.tan(theta_max) ** 2
    b2 = beta ** 2

    def G(c):                                            # primitive of 2 - b2 + b2 c^2
        return (2.0 - b2) * c + b2 * c ** 3 / 3.0

    with np.errstate(divide="ignore", invalid="ignore"):
        # forward: sinh Y + beta c cosh Y > 0  <=>  c > c_fwd
        c_fwd = np.where(beta > 0, -s_y / (beta * c_y), np.where(s_y > 0, -np.inf, np.inf))
        lo = np.clip(c_fwd, -1.0, 1.0)
        # inside the cone: a c^2 + b c + k > 0, i.e. outside the roots (r1, r2)
        a = b2 * (1.0 + t2 * c_y ** 2)
        b = 2.0 * t2 * beta * c_y * s_y
        k = t2 * s_y ** 2 - b2
        disc = b ** 2 - 4.0 * a * k
        root = np.sqrt(np.maximum(disc, 0.0))
        r1 = np.where(disc > 0, (-b - root) / (2.0 * a), 0.0)
        r2 = np.where(disc > 0, (-b + root) / (2.0 * a), 0.0)
    o_lo, o_hi = np.clip(np.maximum(r1, lo), -1.0, 1.0), np.clip(r2, -1.0, 1.0)
    excluded = np.where((disc > 0) & (o_hi > o_lo), G(o_hi) - G(o_lo), 0.0)
    return (G(1.0) - G(lo) - excluded) / (G(1.0) - G(-1.0))


def face_fraction(beta, rapidity, hit_fraction: Callable[[np.ndarray], np.ndarray],
                  distance_m: float, n_c: int = 192) -> np.ndarray:
    """cone_fraction for any face: hit_fraction(r) is the azimuth-averaged
    probability of landing on the face at radius r on the plane at distance_m
    (geometry.hit_fraction), integrated over the decay angle numerically
    (Gauss-Legendre in cos theta*). Vectorised over beta and rapidity."""
    c, w = np.polynomial.legendre.leggauss(n_c)
    b = np.asarray(beta, float).reshape(-1, 1)
    Y = np.asarray(rapidity, float).reshape(-1, 1)
    g = (2.0 - b ** 2) + b ** 2 * c ** 2
    pt = b * np.sqrt(1.0 - c ** 2)
    pz = np.sinh(Y) + b * c * np.cosh(Y)
    fwd = pz > 0
    r = np.where(fwd, distance_m * pt / np.where(fwd, pz, 1.0), np.inf)
    h = np.where(fwd, np.asarray(hit_fraction(r.ravel()), float).reshape(r.shape), 0.0)
    return (g * h * w).sum(axis=1) / (g * w).sum(axis=1)


def acceptance_per_chi(m_chi: float, energy_gev: float, beam_pdf: PDF, target_pdf: PDF,
                       theta_max: Optional[float] = None, *,
                       hit_fraction: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       distance_m: Optional[float] = None, m_pair_min: float = 0.0,
                       n_tau: int = 96, n_y: int = 64) -> float:
    """LO probability that a given chi of the pair lands on the detector,
    fixed target: the pair at zero pT (collinear partons, as in MadGraph LO
    events) with rapidity y + y_cm. Per chi, like the archived acceptance
    files (and the meson channels' hits / 2 N_decays).

    The detector is either a cone of half-angle theta_max around the beam
    (exact, cone_fraction) or any face given by hit_fraction(r) on the plane
    at distance_m (face_fraction; e.g. geometry.hit_fraction for a 1 x 1 m
    face or an off-axis one)."""
    if (theta_max is None) == (hit_fraction is None):
        raise ValueError("give either theta_max or hit_fraction (+ distance_m)")
    s = fixed_target_s(energy_gev)
    d = _differential_pb(m_chi, s, beam_pdf, target_pdf, m_pair_min, ALPHA_EM, n_tau, n_y)
    if d is None:
        return 0.0
    tau, y, dsigma = d
    beta = np.sqrt(np.clip(1.0 - 4.0 * m_chi ** 2 / (tau * s), 0.0, 1.0))
    rapidity = y + fixed_target_y_cm(energy_gev)
    if hit_fraction is None:
        frac = cone_fraction(beta, rapidity, theta_max)
    else:
        frac = face_fraction(beta, rapidity, hit_fraction, distance_m)
    return float(np.sum(dsigma * frac) / np.sum(dsigma))


def rpoint_check() -> float:
    """Reference value: sigma(q qbar -> chi chibar) at s_hat = 100 GeV^2,
    m_chi = 0, eps = 1, e_q = 1 — equals sigma(e+e- -> mu+ mu-)/3 there.
    Used by the test suite as a closed-form anchor."""
    s = 100.0
    return float(4.0 * np.pi * ALPHA_EM ** 2 / (3.0 * s) / N_C * GEV2_TO_PB)
