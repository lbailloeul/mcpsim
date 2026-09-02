"""Drell-Yan (qq-bar -> gamma* -> chi chi-bar) yield from precomputed inputs.

Cross section sigma(m) [pb] (one value per mass, POSITIONALLY row-aligned)
and geometric acceptance ageo(m) (mass, acceptance) come from MadGraph +
detector simulation. Factored from dyProduction (DrawProductionDarkQuest.py).

Normalization: the yield is the Drell-Yan probability per inelastic
proton-nucleon interaction times the POT,

    N(m) = N_POT * sigma_DY(m) / sigma_inel * ageo(m),

with sigma_inel = 13 mb (SIGMA_PN_INELASTIC_MB). The legacy code wrote this
as ``sigma*1e-12 / 13e-3`` with mismatched unit labels; the expression below
is numerically identical (regression-tested) with honest units.

Known data fragility (see the dev-level DY audit and docs/data.md): the cross
file has NO mass column, so pairing with the ageo file is positional. If the
row counts ever differ, dy_yield truncates to the shorter length for legacy
parity and the mismatch is surfaced as a warning upstream. (The shipped
DarkQuest and SHiP pairs are verified row-aligned.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import MB_TO_PB, SIGMA_PN_INELASTIC_MB


def dy_yield(dy_cross_file: str, dy_ageo_file: str, n_pot: float,
             sigma_inelastic_mb: float = SIGMA_PN_INELASTIC_MB,
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline DY yield (N_chi/eps^2) vs mass; returns (masses, yields)."""
    cross = np.loadtxt(dy_cross_file)
    ageo_data = np.loadtxt(dy_ageo_file)

    mass = ageo_data[:, 0]
    ageo = ageo_data[:, 1]
    sigma_pb = cross if cross.ndim == 1 else cross[:, 0]

    # Positional pairing; truncation to the shorter file preserved for
    # legacy parity (the mismatch is reported via row_mismatch()).
    n = min(mass.size, sigma_pb.size, ageo.size)
    mass, sigma_pb, ageo = mass[:n], sigma_pb[:n], ageo[:n]

    yields = n_pot * sigma_pb / (sigma_inelastic_mb * MB_TO_PB) * ageo
    return mass, yields


def row_mismatch(dy_cross_file: str, dy_ageo_file: str) -> Optional[str]:
    """Human-readable description of a cross/ageo row-count mismatch, or None.

    The files are paired by ROW POSITION (the cross file has no mass column);
    unequal lengths mean at least one row is silently mis-paired or dropped."""
    try:
        n_cross = np.atleast_1d(np.loadtxt(dy_cross_file)).shape[0]
        n_ageo = np.atleast_2d(np.loadtxt(dy_ageo_file)).shape[0]
    except OSError:
        return None
    if n_cross == n_ageo:
        return None
    return (f"Drell-Yan input row mismatch: {dy_cross_file} has {n_cross} rows "
            f"but {dy_ageo_file} has {n_ageo}; the files are paired by row "
            f"position, so the extra row(s) are dropped and alignment beyond "
            f"the difference point is suspect. See docs/data.md "
            f"(dev audit: python -m mcpsim.validation.dy_audit).")
