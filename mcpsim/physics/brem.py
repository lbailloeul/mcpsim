"""Proton-bremsstrahlung yield from precomputed (theta, p) cross-section grids.

The Brem_<E>GeV_<mass>.txt files store, per mass, rows of
    log10(theta), log10(p/GeV), sigma@Lambda_p=1.0, @1.5, @2.0   [pb/bin]
over the forward hemisphere. Geometric acceptance is applied at read time as a
theta cut, so the yield can be re-derived for a new detector geometry without
recomputing the cross section (fast mode). A new beam energy needs --regenerate.

Logic factored from brem_single_hit_or_shared_v2 (DrawProductionDarkQuest.py) and
_brem_v2_folder_baseline_single_or_shared (DrawProbability.py).

The target enters twice: the effective luminosity (one interaction length,
rho*lambda/A) and sigma_pA(A) inside the grids. The grids are computed for one
nucleus (declared in their header as `A_<X> = <A> g/mol`), so they are rescaled
by sigma_pA(A_target)/sigma_pA(A_grid) when read for a different target.
Without that, swapping only the luminosity mis-normalises brem (graphite
luminosity on the iron grids: x3.0).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterator, Optional, Tuple

import numpy as np

from .luminosity import effective_lumi_pb

_GRID_A_RE = re.compile(r"^#\s*A_\w+\s*=\s*([0-9.]+(?:[eE][-+]?\d+)?)\s*g/mol")


def sigma_pA_mb(A: float) -> float:
    """sigma_inel - sigma_TSD for p-A [mb] (arXiv:2112.09814 fit), the
    hadronic cross section mCP_brem.py multiplies into every grid entry."""
    return 43.55 * A ** 0.7111 - 3.84 * A ** 0.35


def grid_target_A(path: str) -> Optional[float]:
    """Atomic mass the grid was generated for, from its `# A_<X> = ... g/mol`
    header line; None when the header does not declare it."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _GRID_A_RE.match(line)
            if m:
                return float(m.group(1))
    return None


def target_rescale(path: str, A: float) -> float:
    """sigma_pA(A)/sigma_pA(A_grid) for one grid file (1.0 if undeclared)."""
    A_grid = grid_target_A(path)
    return 1.0 if A_grid is None else sigma_pA_mb(A) / sigma_pA_mb(A_grid)


def _grids(directory: str, pattern: str) -> Iterator[Tuple[float, np.ndarray, str]]:
    """(m_chi, rows, path) for every readable Brem_<E>GeV_<mass>.txt grid;
    files whose name does not end in a mass, or with < 5 columns, are skipped."""
    for f in sorted(glob.glob(os.path.join(directory, pattern))):
        try:
            mchi = float(os.path.basename(f).split("_")[-1].replace(".txt", ""))
        except ValueError:
            continue
        arr = np.loadtxt(f, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] >= 5:
            yield mchi, arr, f


def _by_mass(masses: list, yields: list) -> Tuple[np.ndarray, np.ndarray]:
    masses, yields = np.asarray(masses), np.asarray(yields)
    order = np.argsort(masses)
    return masses[order], yields[order]


def brem_yield(directory: str, pattern: str, det_angle: float,
               n_pot: float, density_g_cm3: float, interaction_length_cm: float,
               A: float, lambda_idx: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline bremsstrahlung yield (events at epsilon = 1) vs mass.

    Returns (masses, yields), sorted by mass. `det_angle` is the polar
    acceptance half-angle (cylindrical: arctan(R/L); bar-array: an equal-area
    equivalent, see geometry.equivalent_theta_cut). lambda_idx selects the
    off-shell form-factor cutoff (0->1.0, 1->1.5, 2->2.0 GeV). Each grid is
    rescaled to the target's sigma_pA (target_rescale).
    """
    if lambda_idx not in (0, 1, 2):
        raise ValueError("lambda_idx must be 0, 1, or 2")

    L_eff = effective_lumi_pb(n_pot, density_g_cm3, interaction_length_cm, A)
    log10_theta_cut = np.log10(det_angle)

    masses, yields = [], []
    for mchi, arr, f in _grids(directory, pattern):
        mask = arr[:, 0] < log10_theta_cut
        sigma_sum_pb = arr[:, 2:5][mask, lambda_idx].sum()
        masses.append(mchi)
        yields.append(L_eff * target_rescale(f, A) * sigma_sum_pb)
    return _by_mass(masses, yields)


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
    (sigma is per-bin; distributed uniformly in log theta). Each grid is
    rescaled to the target's sigma_pA (target_rescale).
    """
    from .. import geometry

    if lambda_idx not in (0, 1, 2):
        raise ValueError("lambda_idx must be 0, 1, or 2 (Lambda_p = 1.0/1.5/2.0 GeV)")

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
    for mchi, arr, f in _grids(directory, pattern):
        w = _weights(arr[:, 0])
        masses.append(mchi)
        yields.append(L_eff * target_rescale(f, A) * float((arr[:, 2 + lambda_idx] * w).sum()))
    return _by_mass(masses, yields)


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
