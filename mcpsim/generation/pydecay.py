"""Vectorized Python decay engine: meson -> chi chi-bar acceptance.

Re-decays archived PYTHIA parent mesons (generation/samples.py) to
millicharged pairs and scores each chi with the azimuthal-average kernel
(geometry.face_hit_fraction / circle_hit_fraction) on the detector plane —
acceptance for ANY geometry, including off-axis faces, in minutes and with no
ROOT/PYTHIA toolchain. Promoted from the FLAME off-axis study
(scripts/flame_offaxis.py) and validated against the legacy C++ decay binaries
(pytest -m parity).

Conventions (identical to lanl_decayPion_12bar.cc / decayVectorMeson.cc):
  * acceptance = per-chi hit probability, hits / (2 * N_decays); the factor
    2 chis/decay, branching ratios and I2/I3 phase space are applied
    downstream in physics/meson_decay.meson_yield.
  * two-body decays use the per-event Breit-Wigner mother mass and drop
    sub-threshold parents from numerator AND denominator (matters for the
    150 MeV-wide rho).

Fidelity:
  * corrected (default) — Dalitz gamma* direction isotropic in the meson
    frame, chi polar angle (2 - xi sin^2 theta) about the gamma* boost axis.
  * legacy — replicates the C++ 'dalitz' sampler quirks bit-for-bit in
    method (theta_V uniform in [0, pi], chi built about the z-axis, boosted
    along the tilted direction unrotated). Differences are <= +-8% in ageo
    and <~1% on the epsilon limit (see validation Test C).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .. import geometry
from ..config import Config, DetectorConfig
from ..physics.constants import MESONS, MesonSpec
from . import samples

MAX_SCAN_MASSES = 33          # acceptance is smooth in m_chi; interp fills the grid
CHUNK_PAIRS = 2_000_000       # decayed pairs per numpy chunk (memory bound)


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------
def boost(px, py, pz, E, bx, by, bz, gamma):
    """Active boost of (E, p) by velocity (bx, by, bz), gamma = 1/sqrt(1-b^2)."""
    b2 = bx * bx + by * by + bz * bz
    pdotb = px * bx + py * by + pz * bz
    coef = np.where(b2 > 0, (gamma - 1.0) * pdotb / np.where(b2 > 0, b2, 1.0), 0.0) \
        + gamma * E
    return (px + coef * bx, py + coef * by, pz + coef * bz,
            gamma * (E + pdotb))


def isotropic_dirs(rng: np.random.Generator, n: int):
    u = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - u * u)
    return s * np.cos(ph), s * np.sin(ph), u


def sample_dalitz_s(rng: np.random.Generator, mchi: float, mother: float,
                    n: int) -> np.ndarray:
    """Invariant mass^2 of the chi pair, marginal over angles (log-s CDF).

    Marginal weight in t = ln(s): (1 - s/M^2)^3 sqrt(xi) (1 - xi/3), the
    angular integral of the (2 - xi sin^2 theta) Dalitz weight (matches
    marginalLogSWeight in lanl_decayPion_12bar.cc)."""
    t = np.linspace(np.log(4.0 * mchi * mchi) + 1e-12,
                    np.log(mother * mother) - 1e-12, 3000)
    s = np.exp(t)
    xi = 1.0 - 4.0 * mchi * mchi / s
    w = (1.0 - s / mother**2) ** 3 * np.sqrt(np.clip(xi, 0, None)) * (1.0 - xi / 3.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    return np.exp(np.interp(rng.uniform(0, 1, n), cdf, t))


def sample_costheta_rel(rng: np.random.Generator, xi: np.ndarray) -> np.ndarray:
    """cos(theta) of the chi in the gamma* frame, density ∝ (2 - xi) + xi u^2
    (the 2 - xi sin^2 theta Dalitz weight), by vectorized rejection."""
    u = np.empty_like(xi)
    todo = np.ones(xi.shape, dtype=bool)
    while np.any(todo):
        n = int(todo.sum())
        prop = rng.uniform(-1.0, 1.0, n)
        acc = rng.uniform(0.0, 2.0, n) <= (2.0 - xi[todo]) + xi[todo] * prop * prop
        idx = np.flatnonzero(todo)[acc]
        u[idx] = prop[acc]
        todo[idx] = False
    return u


def decay_to_radii(spec: MesonSpec, mchi: float, P, rng: np.random.Generator,
                   distance_m: float, fidelity: str = "corrected"
                   ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decay parent 4-momenta P = (px,py,pz,E) -> landing radii of both chis
    on the z = distance_m plane. Returns (r1, r2, n_kept); radii are NaN for
    backward chis and for parents below the 2 m_chi threshold.

    Two-body channels use the per-event Breit-Wigner mother mass and drop
    sub-threshold parents from numerator AND denominator, matching
    decayVectorMeson.cc (identical in both fidelity modes)."""
    px, py, pz, E = P
    n = px.size
    M = spec.mass

    if spec.channel == "dalitz":
        s = sample_dalitz_s(rng, mchi, M, n)
        xi = 1.0 - 4.0 * mchi * mchi / s
        u = sample_costheta_rel(rng, xi)
        pstar = np.sqrt(np.clip(s / 4.0 - mchi * mchi, 0, None))
        sr = np.sqrt(np.clip(1.0 - u * u, 0, None))
        phr = rng.uniform(0, 2 * np.pi, n)
        Ec = np.hypot(pstar, mchi)
        E_V = (M * M + s) / (2.0 * M)
        q = (M * M - s) / (2.0 * M)
        bV = q / E_V
        gV = E_V / np.sqrt(s)

        if fidelity == "legacy":
            # C++ 'dalitz' branch: chi about the z-axis, tilted boost,
            # theta_V flat in [0, pi] (pole-biased). Kept for parity runs.
            cx = pstar * sr * np.cos(phr)
            cy = pstar * sr * np.sin(phr)
            cz = pstar * u
            tV = rng.uniform(0.0, np.pi, n)
            pV = rng.uniform(0.0, 2 * np.pi, n)
            nx, ny, nz = (np.sin(tV) * np.cos(pV), np.sin(tV) * np.sin(pV),
                          np.cos(tV))
        else:
            # corrected: isotropic gamma* direction, theta_rel about the
            # boost axis (what the C++ 'paper' mode intends).
            nx, ny, nz = isotropic_dirs(rng, n)
            st = np.sqrt(np.clip(1.0 - nz * nz, 1e-30, None))
            e1x, e1y, e1z = -ny / st, nx / st, np.zeros(n)
            e2x = ny * e1z - nz * e1y
            e2y = nz * e1x - nx * e1z
            e2z = nx * e1y - ny * e1x
            cx = pstar * (sr * (np.cos(phr) * e1x + np.sin(phr) * e2x) + u * nx)
            cy = pstar * (sr * (np.cos(phr) * e1y + np.sin(phr) * e2y) + u * ny)
            cz = pstar * (sr * (np.cos(phr) * e1z + np.sin(phr) * e2z) + u * nz)

        c1 = boost(cx, cy, cz, Ec, bV * nx, bV * ny, bV * nz, gV)
        c2 = boost(-cx, -cy, -cz, Ec, bV * nx, bV * ny, bV * nz, gV)
        Mev = np.full(n, M)
        kept = np.ones(n, dtype=bool)
    else:  # two_body: isotropic in the meson rest frame, per-event BW mass
        Mev = np.sqrt(np.clip(E * E - (px * px + py * py + pz * pz), 1e-12, None))
        kept = Mev > 2.0 * mchi
        pstar = np.sqrt(np.clip(Mev * Mev / 4.0 - mchi * mchi, 0, None))
        nx, ny, nz = isotropic_dirs(rng, n)
        cx, cy, cz = pstar * nx, pstar * ny, pstar * nz
        Ec = np.sqrt(pstar**2 + mchi**2)
        c1 = (cx, cy, cz, Ec)
        c2 = (-cx, -cy, -cz, Ec)

    gL = E / Mev
    out = []
    for c in (c1, c2):
        lx, ly, lz, _ = boost(*c, px / E, py / E, pz / E, gL)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((lz > 0) & kept, distance_m * np.hypot(lx, ly) / lz,
                         np.nan)
        out.append(r)
    return out[0], out[1], int(kept.sum())


# --------------------------------------------------------------------------
# Acceptance scan
# --------------------------------------------------------------------------
def _hit_fraction(det: DetectorConfig, r: np.ndarray) -> np.ndarray:
    d = geometry.offaxis_offset_m(det)
    if det.type == "bar_array":
        half_w = 0.5 * det.bar_columns * det.bar_size_m
        half_h = 0.5 * det.bar_rows * det.bar_size_m
        return geometry.face_hit_fraction(r, d, half_w, half_h)
    if det.type == "cylindrical":
        return geometry.circle_hit_fraction(r, d, det.radius_m)
    raise ValueError(f"unknown detector type {det.type!r}")


def compute_acceptance(cfg: Config, meson: str, grid: np.ndarray, *,
                       quick: Optional[bool] = None, verbose: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-chi geometric acceptance vs mass for one meson.

    Returns (scan_masses, ageo, relerr): the thinned mass scan (interpolate
    onto the full grid downstream), the acceptance, and the clustered MC
    relative error (accounts for re-decaying each parent `repeats` times).
    """
    if quick is None:
        quick = cfg.engine.quick
    spec = MESONS[meson]
    parents, repeats = samples.load_parents(cfg, meson, quick=quick)
    n_parents = parents[0].size
    rng = np.random.default_rng(cfg.engine.seed)

    below = np.flatnonzero(grid < spec.mass / 2.0)
    k = min(below.size, MAX_SCAN_MASSES if not quick else 9)
    scan_idx = below[np.unique(np.round(np.linspace(0, below.size - 1, k)).astype(int))]
    scan_m = grid[scan_idx]

    nm = scan_m.size
    S1 = np.zeros(nm)
    Sc2 = np.zeros(nm)               # clustered-by-parent sum of squares
    n_kept = np.zeros(nm)
    chunk_parents = max(1, CHUNK_PAIRS // repeats)
    t0 = time.time()

    for j, mchi in enumerate(scan_m):
        for lo in range(0, n_parents, chunk_parents):
            hi = min(lo + chunk_parents, n_parents)
            Pc = tuple(np.tile(a[lo:hi], repeats) for a in parents)
            r1, r2, nk = decay_to_radii(spec, mchi, Pc, rng,
                                        cfg.detector.distance_m,
                                        cfg.engine.fidelity)
            n_kept[j] += nk
            f = (np.nan_to_num(_hit_fraction(cfg.detector, r1))
                 + np.nan_to_num(_hit_fraction(cfg.detector, r2)))
            S1[j] += f.sum()
            cl = f.reshape(repeats, hi - lo).sum(axis=0)
            Sc2[j] += (cl * cl).sum()

    ageo = S1 / (2.0 * np.maximum(n_kept, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        relerr = np.where(S1 > 0, np.sqrt(Sc2) / S1, np.nan)
    if verbose:
        worst = np.nanmax(relerr) if np.any(np.isfinite(relerr)) else float("nan")
        print(f"[mcpsim] pydecay {meson:6s} parents={n_parents:>9,} "
              f"rep={repeats:<4d} masses={nm:<3d} {time.time()-t0:6.1f}s  "
              f"worst relerr={worst:.2%}")
    return scan_m, ageo, relerr


def acceptance_on_grid(cfg: Config, meson: str, grid: np.ndarray, *,
                       quick: Optional[bool] = None) -> np.ndarray:
    """Acceptance interpolated onto the full mass grid (zero beyond threshold)."""
    spec = MESONS[meson]
    scan_m, ageo, _ = compute_acceptance(cfg, meson, grid, quick=quick)
    ag = np.interp(grid, scan_m, ageo, left=ageo[0] if ageo.size else 0.0,
                   right=0.0)
    ag[grid >= spec.mass / 2.0] = 0.0
    return ag


def geometry_hash(cfg: Config, meson: str, grid: np.ndarray) -> str:
    """Cache key: everything the acceptance result depends on."""
    det = cfg.detector
    spec = samples.DEFAULT_SAMPLES.get(meson)
    payload = {
        "det": [det.type, det.distance_m, det.radius_m, det.bar_columns,
                det.bar_rows, det.bar_size_m, det.offaxis_mrad],
        "engine": [cfg.engine.fidelity, cfg.engine.seed,
                   str(samples.samples_dir(cfg))],
        "sample": [spec.file, spec.n_use, spec.repeats] if spec else None,
        "grid": [float(grid[0]), float(grid[-1]), int(grid.size)],
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cached_acceptance(cfg: Config, meson: str, grid: np.ndarray, cache_dir: Path,
                      *, quick: Optional[bool] = None) -> np.ndarray:
    """acceptance_on_grid with an on-disk cache + JSON provenance sidecar."""
    if quick is None:
        quick = cfg.engine.quick
    key = geometry_hash(cfg, meson, grid)
    sub = cache_dir / key
    out = sub / f"ageo_{meson}.txt"
    if out.exists() and not quick:
        arr = np.loadtxt(out)
        if arr.shape[0] == grid.size:
            return arr[:, 1]
    sub.mkdir(parents=True, exist_ok=True)
    ag = acceptance_on_grid(cfg, meson, grid, quick=quick)
    if not quick:
        np.savetxt(out, np.column_stack([grid, ag]), fmt="%.8g",
                   header=f"mass_GeV ageo(per-chi hit prob) [pydecay {key}]")
        det = cfg.detector
        with open(sub / f"ageo_{meson}.json", "w") as fh:
            json.dump({"meson": meson, "detector": {
                "type": det.type, "distance_m": det.distance_m,
                "radius_m": det.radius_m,
                "bars": [det.bar_columns, det.bar_rows, det.bar_layers],
                "bar_size_m": det.bar_size_m, "offaxis_mrad": det.offaxis_mrad},
                "fidelity": cfg.engine.fidelity, "seed": cfg.engine.seed,
                "samples_dir": str(samples.samples_dir(cfg))}, fh, indent=2)
    return ag
