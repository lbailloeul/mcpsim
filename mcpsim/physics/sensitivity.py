"""Charge (epsilon) dependence and exclusion-contour assembly.

The detection factor models the light-yield / scattering response used in
DrawProbability.py and the LANL sensitivity script:

    D(eps) = (1 - exp(-N_gamma * eps^2))^a * eps^2

A baseline yield (N_chi at eps = 1, i.e. N_chi/eps^2) is broadcast against a grid
of charges to a 2D signal array N_chi(eps, m). The exclusion contour is the
N_chi = threshold level set.
"""

from __future__ import annotations

import numpy as np


def detection_factor(charges: np.ndarray, n_gamma: float, a: float) -> np.ndarray:
    """D(eps) = (1 - exp(-N_gamma eps^2))^a * eps^2."""
    charges = np.asarray(charges, dtype=float)
    return (1.0 - np.exp(-n_gamma * charges ** 2)) ** a * charges ** 2


def signal_2d(baseline_1d: np.ndarray, charges: np.ndarray,
              n_gamma: float, a: float) -> np.ndarray:
    """Broadcast a 1D baseline yield to 2D N_chi(eps, m).

    Returns shape (n_charges, n_masses).
    """
    det = detection_factor(charges, n_gamma, a)
    baseline_1d = np.asarray(baseline_1d, dtype=float)
    return det[:, None] * baseline_1d[None, :]


def charge_grid(charge_min: float, charge_max: float, n_charge: int) -> np.ndarray:
    """Log-spaced epsilon scan grid (strictly increasing, validated).

    A swapped min/max would make exclusion_contour silently return the
    LARGEST charge at every mass — validate instead."""
    if not (0 < charge_min < charge_max):
        raise ValueError(
            f"sensitivity charge grid needs 0 < charge_min < charge_max, got "
            f"charge_min={charge_min!r}, charge_max={charge_max!r}"
        )
    if n_charge < 2:
        raise ValueError(f"n_charge must be >= 2, got {n_charge}")
    return np.logspace(np.log10(charge_min), np.log10(charge_max), n_charge)


def exclusion_contour(baseline_1d: np.ndarray, charges: np.ndarray,
                      n_gamma: float, a: float, threshold: float) -> np.ndarray:
    """Smallest epsilon at which the signal reaches `threshold`, per mass.

    The signal N_chi(eps, m) = D(eps) * baseline(m) is monotone increasing in
    eps, so the exclusion region is eps > eps95(m). The crossing is found per
    mass column and refined by log-log interpolation between the bracketing
    grid points. Masses where the signal never reaches the threshold (or where
    already the smallest eps exceeds it) get NaN / charges[0] respectively.

    This is the single contour solver shared by the limit plot and the
    text-table outputs (and previously duplicated in scripts/flame_offaxis.py).
    """
    baseline_1d = np.asarray(baseline_1d, dtype=float)
    signal = signal_2d(baseline_1d, charges, n_gamma, a)     # (n_eps, n_mass)
    lim = np.full(baseline_1d.size, np.nan)
    for j in range(baseline_1d.size):
        above = np.flatnonzero(signal[:, j] >= threshold)
        if above.size == 0:
            continue
        k = above[0]
        if k == 0:
            lim[j] = charges[0]
            continue
        x0, x1 = np.log(signal[k - 1, j]), np.log(signal[k, j])
        e0, e1 = np.log(charges[k - 1]), np.log(charges[k])
        lim[j] = np.exp(e0 + (np.log(threshold) - x0) * (e1 - e0) / (x1 - x0))
    return lim
