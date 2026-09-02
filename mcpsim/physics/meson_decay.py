"""Meson decay into millicharged pairs (2-body and 3-body Dalitz).

Phase-space integrals and yields, factored verbatim from
DrawProductionDarkQuest.py / DrawProbability.py so results are identical.

A 'baseline' yield is N_chi at epsilon = 1, i.e. the N_chi/epsilon^2 quantity
plotted on the production axis. The epsilon / detection dependence is applied
separately in sensitivity.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import quad

from .constants import ALPHA_EM, M_ELECTRON, MesonSpec


def I2(x: float, y: float) -> float:
    """2-body phase-space ratio (x = (m_chi/M)^2, y = (m_e/M)^2)."""
    return ((1 + 2 * x) * np.sqrt(1 - 4 * x)) / ((1 + 2 * y) * np.sqrt(1 - 4 * y))


def _I3_integrand(z: float, x: float, pi_val: float) -> float:
    # pi_val: the legacy scripts used 3.14 (not np.pi); fidelity='legacy'
    # keeps that for bit-for-bit parity (+0.051% on every Dalitz yield).
    return 2 / (3 * pi_val) * np.sqrt(1 - 4 * x / z) * ((1 - z) ** 3) * (2 * x + z) / (z ** 2)


def I3(x: float, pi_val: float = np.pi) -> float:
    """3-body (Dalitz) phase-space integral."""
    return quad(_I3_integrand, 4 * x, 1, args=(x, pi_val))[0]


_vI3 = np.vectorize(I3)


def meson_yield(meson: MesonSpec, masses: np.ndarray, ageo: np.ndarray,
                n_pot: float, c_meson: float,
                fidelity: str = "corrected") -> np.ndarray:
    """Baseline mCP yield (N_chi/eps^2) for one meson channel on `masses`.

    masses, ageo are aligned arrays (ageo is the geometric acceptance per mass).
    Yields are zero where 2*m_chi exceeds the meson mass. fidelity='legacy'
    reproduces the published pipeline's 3.14-instead-of-pi Dalitz constant.
    """
    masses = np.asarray(masses, dtype=float)
    ageo = np.asarray(ageo, dtype=float)
    out = np.zeros_like(masses)

    open_ = masses < meson.mass / 2.0
    if not np.any(open_):
        return out

    x = masses[open_] ** 2 / meson.mass ** 2
    if meson.channel == "dalitz":
        pi_val = 3.14 if fidelity == "legacy" else np.pi
        out[open_] = (n_pot * ageo[open_] * 2 * c_meson * meson.branching
                      * ALPHA_EM * _vI3(x, pi_val))
    elif meson.channel == "two_body":
        y = M_ELECTRON ** 2 / meson.mass ** 2
        out[open_] = (n_pot * ageo[open_] * 2 * c_meson * meson.branching
                      * I2(x, y))
    else:  # pragma: no cover - guarded by MesonSpec construction
        raise ValueError(f"unknown decay channel {meson.channel!r}")
    return out
