"""Analytic LO Drell-Yan partonic cross section (the MadGraph-free tier of
the DY validation).

    sigma(q qbar -> gamma* -> chi chibar)
      = (4 pi alpha^2 / 3 s_hat) * Q_q^2 * eps^2 * (1/N_c)
        * beta * (1 + 2 m_chi^2 / s_hat) / normalization of massless leptons,
    beta = sqrt(1 - 4 m_chi^2 / s_hat)

The 1/N_c = 1/3 averages the initial quark colors. At eps = 1, m_chi -> 0 and
Q_q = 1 this reduces to the textbook sigma(e+e- -> mu+ mu-) = 4 pi alpha^2 /
(3 s) divided by 3 — the standard qqbar -> l+l- result. Convolving with PDFs
(LHAPDF, if available) gives the hadronic cross section for comparison with
the archived MadGraph scans; the partonic formula alone anchors the UFO-model
muon-pair benchmark (see models/mcp_ufo/__init__.py).
"""

from __future__ import annotations

import numpy as np

from .constants import ALPHA_EM

GEV2_TO_PB = 3.893793721e8   # (hbar c)^2 in GeV^2 * pb
N_C = 3.0


def sigma_partonic_pb(s_hat_gev2, m_chi: float, eps: float = 1.0,
                      e_q: float = 1.0) -> np.ndarray:
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
    x[open_] = (4.0 * np.pi * ALPHA_EM ** 2 / (3.0 * ss)
                * e_q ** 2 * eps ** 2 / N_C
                * beta * (1.0 + 2.0 * m_chi ** 2 / ss))
    return x * GEV2_TO_PB


def rpoint_check() -> float:
    """Reference value: sigma(q qbar -> chi chibar) at s_hat = 100 GeV^2,
    m_chi = 0, eps = 1, e_q = 1 — equals sigma(e+e- -> mu+ mu-)/3 there.
    Used by the test suite as a closed-form anchor."""
    s = 100.0
    return float(4.0 * np.pi * ALPHA_EM ** 2 / (3.0 * s) / N_C * GEV2_TO_PB)
