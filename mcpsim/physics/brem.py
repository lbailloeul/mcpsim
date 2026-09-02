"""Proton-bremsstrahlung yield from precomputed (theta, p) cross-section grids.

The Brem_<E>GeV_<mass>.txt files store, per mass, rows of
    log10(theta), log10(p/GeV), sigma@Lambda_p=1.0, @1.5, @2.0   [pb/bin]
over the forward hemisphere. Geometric acceptance is applied at read time as a
theta cut, so the yield can be re-derived for a new detector geometry without
recomputing the cross section (fast mode). A new beam energy needs --regenerate.

Logic factored from brem_single_hit_or_shared_v2 (DrawProductionDarkQuest.py) and
_brem_v2_folder_baseline_single_or_shared (DrawProbability.py).
"""

from __future__ import annotations

import glob
import os
from typing import Tuple

import numpy as np

from .luminosity import effective_lumi_pb

LAMBDA_VALUES = (1.0, 1.5, 2.0)  # column index 0/1/2 in the sigma block


def brem_yield(directory: str, pattern: str, det_angle: float,
               n_pot: float, density_g_cm3: float, interaction_length_cm: float,
               A: float, lambda_idx: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline bremsstrahlung yield (events at epsilon = 1) vs mass.

    Returns (masses, yields), sorted by mass. `det_angle` is the polar
    acceptance half-angle (cylindrical: arctan(R/L); bar-array: an equal-area
    equivalent, see geometry.equivalent_theta_cut). lambda_idx selects the
    off-shell form-factor cutoff (0->1.0, 1->1.5, 2->2.0 GeV).
    """
    if lambda_idx not in (0, 1, 2):
        raise ValueError("lambda_idx must be 0, 1, or 2")

    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return np.array([]), np.array([])

    L_eff = effective_lumi_pb(n_pot, density_g_cm3, interaction_length_cm, A)
    log10_theta_cut = np.log10(det_angle)

    masses, yields = [], []
    for f in files:
        try:
            mchi = float(os.path.basename(f).split("_")[-1].replace(".txt", ""))
        except ValueError:
            continue

        arr = np.loadtxt(f, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 5:
            continue

        log10theta = arr[:, 0]
        sigmas = arr[:, 2:5]
        mask = log10theta < log10_theta_cut
        sigma_sum_pb = sigmas[mask, lambda_idx].sum()

        masses.append(mchi)
        yields.append(L_eff * sigma_sum_pb)

    masses = np.asarray(masses)
    yields = np.asarray(yields)
    order = np.argsort(masses)
    return masses[order], yields[order]


def brem_yield_offaxis(directory: str, pattern: str, det, n_pot: float,
                       density_g_cm3: float, interaction_length_cm: float,
                       A: float, lambda_idx: int = 1,
                       n_sub: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline brem yield through a displaced (or exact rectangular) face.

    Instead of the on-axis cone cut, each (theta, p) bin is weighted by the
    azimuthal fraction of its ring on the z = distance plane that intersects
    the face (geometry.face_hit_fraction / circle_hit_fraction). The theta
    grid is coarse (0.045 dex) next to the radial band a small face subtends,
    so each bin is smeared over its log10-theta width with `n_sub` slices
    (sigma is per-bin; distributed uniformly in log theta).
    """
    from .. import geometry

    if lambda_idx not in (0, 1, 2):
        raise ValueError("lambda_idx must be 0, 1, or 2 (Lambda_p = 1.0/1.5/2.0 GeV)")

    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return np.array([]), np.array([])

    L_eff = effective_lumi_pb(n_pot, density_g_cm3, interaction_length_cm, A)
    d = geometry.offaxis_offset_m(det)

    def _weights(log10theta: np.ndarray) -> np.ndarray:
        ult, inv = np.unique(log10theta, return_inverse=True)
        half = 0.5 * (np.median(np.diff(ult)) if ult.size > 1 else 0.045)
        sub = np.linspace(-half, half, n_sub)
        th = 10.0 ** (ult[:, None] + sub[None, :])
        r = np.where(th < np.pi / 2, det.distance_m * np.tan(th), np.inf)
        if det.type == "bar_array":
            half_w = 0.5 * det.bar_columns * det.bar_size_m
            half_h = 0.5 * det.bar_rows * det.bar_size_m
            w = geometry.face_hit_fraction(r.ravel(), d, half_w, half_h)
        else:
            w = geometry.circle_hit_fraction(r.ravel(), d, det.radius_m)
        return w.reshape(th.shape).mean(axis=1)[inv]

    masses, yields = [], []
    for f in files:
        try:
            mchi = float(os.path.basename(f).split("_")[-1].replace(".txt", ""))
        except ValueError:
            continue
        arr = np.loadtxt(f, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 5:
            continue
        w = _weights(arr[:, 0])
        masses.append(mchi)
        yields.append(L_eff * float((arr[:, 2 + lambda_idx] * w).sum()))

    masses = np.asarray(masses)
    yields = np.asarray(yields)
    order = np.argsort(masses)
    return masses[order], yields[order]


def interp_to_grid(masses: np.ndarray, yields: np.ndarray,
                   grid: np.ndarray) -> np.ndarray:
    """Interpolate a brem yield curve onto a target mass grid (log-log).

    Masses outside the computed brem range are zero-filled (no extrapolation):
    proton bremsstrahlung does not produce pairs beyond the tabulated mass range,
    so holding the endpoint constant would spuriously inflate the high-mass yield.
    """
    grid = np.asarray(grid, dtype=float)
    out = np.zeros_like(grid)
    if masses.size == 0:
        return out
    pos = (yields > 0) & (masses > 0)
    if not np.any(pos):
        return out
    xm, ym = np.log10(masses[pos]), np.log10(yields[pos])
    inrange = grid > 0
    yq = np.interp(np.log10(grid[inrange]), xm, ym, left=-np.inf, right=-np.inf)
    valid = np.isfinite(yq)
    out[np.where(inrange)[0][valid]] = 10.0 ** yq[valid]
    return out
