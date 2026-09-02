"""Effective beam-dump luminosity.

Production is counted only in the first nuclear interaction length, following the
DarkQuest / SHiP treatment (mCP_brem.py, DrawProduction*).
"""

from __future__ import annotations

from .constants import N_AVOGADRO


def effective_lumi_pb(n_pot: float, density_g_cm3: float,
                      interaction_length_cm: float, A: float) -> float:
    """Effective luminosity in pb^-1.

        L_eff = N_POT * (rho * L / A) * N_A          [cm^-2]
              / 1e36                                  [-> pb^-1]
    """
    nuclei_per_cm2 = (density_g_cm3 * interaction_length_cm / A) * N_AVOGADRO
    return n_pot * nuclei_per_cm2 / 1e36
