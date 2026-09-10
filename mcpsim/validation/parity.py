"""Parity harness: the Python decay engine vs the legacy C++ decay binaries.

Three tests, promoted from scripts/validate_decay_parity.py:

  A (2-body): decayVectorMeson.cc on the archived rho sample with its
     hardcoded 100 m / 0.5 m cone vs pydecay.decay_to_radii + the equivalent
     radius cut on the SAME parents. Validates the per-event-BW-mass 2-body
     model; expected agreement within ~1 sigma.
  B (Dalitz): lanl_decayPion_12bar.cc ('dalitz' sampler) on a text pi0 sample
     vs pydecay in LEGACY fidelity (replicating the C++ sampler exactly) —
     the implementation check — and CORRECTED fidelity for reference.
  C: corrected/legacy Dalitz acceptance ratio in an off-axis FLAME-like
     geometry at 2% kernel precision — sizes the deliberate physics fix.

Everything runs from the archived samples; needs ROOT (compiles the C++) and
the source repo. Scratch space: <WORK_DIR>/validation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import paths
from ..config import Config, DetectorConfig, EngineConfig
from ..generation import pydecay, samples
from ..generation.common import compile_cpp
from ..physics.constants import MESONS

SCRATCH = paths.WORK_DIR / "validation"

RHO_MASSES = [0.01, 0.10, 0.30]          # < m_rho/2
PI0_MASSES = [0.005, 0.02, 0.05]         # < m_pi0/2
N_PI0 = 4_000_000
PI0_MOTHER = 0.1349768                    # the C++ binary's mother-mass argument

CONE_D, CONE_R = 100.0, 0.5               # decayVectorMeson.cc hardcoded geometry
SIDE_D, SIDE_HALF = 6.0, 0.25             # 10x10 bars of 5 cm -> 0.5 m face


@dataclass
class ParityRow:
    test: str
    m_chi: float
    eff_cpp: float
    eff_py: float
    z: float                  # (cpp - py) / combined stat error
    ratio: float
    extra: Optional[float] = None   # test B: corrected-mode efficiency


def _cfg(quick: bool = False) -> Config:
    return Config(engine=EngineConfig(quick=quick))


def _eff_err(hits: float, total: float) -> float:
    return np.sqrt(max(hits, 1.0)) / total


# ---------------------------------------------------------------- Test A ----
def test_2body(quick: bool = False, verbose: bool = True) -> List[ParityRow]:
    """decayVectorMeson.cc vs pydecay on identical rho parents + cone cut."""
    import uproot

    SCRATCH.mkdir(parents=True, exist_ok=True)
    binary = compile_cpp(paths.legacy_dir() / "decayVectorMeson.cc",
                        paths.CACHE_DIR / "bin" / "decayVectorMeson")
    cfg = _cfg(quick)
    rho = samples.samples_dir(cfg) / samples.DEFAULT_SAMPLES["rho"].file
    arr = uproot.open(rho)["mesons"].arrays(["px", "py", "pz", "e"], library="np")
    P = tuple(arr[k].astype(np.float64) for k in ("px", "py", "pz", "e"))
    theta_cut = np.arctan2(CONE_R, CONE_D)
    rng = np.random.default_rng(cfg.engine.seed)
    spec = MESONS["rho"]

    rows = []
    masses = RHO_MASSES[:1] if quick else RHO_MASSES
    for m in masses:
        out = SCRATCH / f"vm_{m:g}.root"
        with open(SCRATCH / f"vm_{m:g}.log", "w") as log:
            subprocess.run([str(binary), str(rho), str(out), str(m),
                            str(SCRATCH / f"vm_eff_{m:g}.txt")],
                           check=True, stdout=log, stderr=log)
        with uproot.open(out) as f:
            c_hits = f["mcp-filtered"].num_entries
            c_tot = f["mcp"].num_entries
        c_eff, c_err = c_hits / c_tot, _eff_err(c_hits, c_tot)

        # cone theta <= cut is a radius cut on any projection plane
        r_cut = 1000.0 * np.tan(theta_cut)
        r1, r2, nk = pydecay.decay_to_radii(spec, m, P, rng, 1000.0, "corrected")
        p_hits = int((np.nan_to_num(r1, nan=np.inf) <= r_cut).sum()
                     + (np.nan_to_num(r2, nan=np.inf) <= r_cut).sum())
        p_eff, p_err = p_hits / (2 * nk), _eff_err(p_hits, 2 * nk)

        z = (c_eff - p_eff) / np.hypot(c_err, p_err)
        rows.append(ParityRow("A:2body", m, c_eff, p_eff, float(z),
                              c_eff / p_eff if p_eff else float("inf")))
        if verbose:
            print(f"  A m={m:6.3g}  cpp={c_eff:.4e}  py={p_eff:.4e}  "
                  f"ratio={c_eff/max(p_eff,1e-30):.3f}  z={z:+.1f}")
    return rows


# ---------------------------------------------------------------- Test B ----
def _pi0_parents(cfg: Config, n: int):
    import uproot

    t = uproot.open(samples.samples_dir(cfg)
                    / samples.DEFAULT_SAMPLES["pi0"].file)["mesons"]
    a = t.arrays(["magnitude", "theta", "phi"], entry_stop=n, library="np")
    p = a["magnitude"].astype(np.float64)
    ct = np.cos(a["theta"].astype(np.float64))
    ph = a["phi"].astype(np.float64)
    txt = SCRATCH / f"pi0_source_{n}.txt"
    if not txt.exists():
        np.savetxt(txt, np.column_stack([p * 1e3, ct, ph]), fmt="%.8g")
    st = np.sqrt(np.clip(1 - ct * ct, 0, None))
    E = np.hypot(p, PI0_MOTHER)
    return txt, (p * st * np.cos(ph), p * st * np.sin(ph), p * ct, E)


def _side_plane_hits(P, m, fidelity, rng) -> int:
    """Decay pi0 parents and count chis through the LANL-style side plane.

    The C++ models the detector as the plane x = SIDE_D; equivalently, decay
    with the beam along z and count lx > 0 with |SIDE_D*ly/lx|, |SIDE_D*lz/lx|
    inside the half-face."""
    spec = replace(MESONS["pi0"], mass=PI0_MOTHER)
    # decay_to_radii projects on a z-plane; for the side plane we need the raw
    # lab momenta, so replicate its Dalitz stage via a large-distance trick is
    # not enough — use the internal sampler directly.
    px, py, pz, E = P
    n = px.size
    s = pydecay.sample_dalitz_s(rng, m, PI0_MOTHER, n)
    xi = 1.0 - 4.0 * m * m / s
    u = pydecay.sample_costheta_rel(rng, xi)
    pstar = np.sqrt(np.clip(s / 4.0 - m * m, 0, None))
    sr = np.sqrt(np.clip(1.0 - u * u, 0, None))
    phr = rng.uniform(0, 2 * np.pi, n)
    Ec = np.hypot(pstar, m)
    E_V = (PI0_MOTHER ** 2 + s) / (2 * PI0_MOTHER)
    bV = (PI0_MOTHER ** 2 - s) / (2 * PI0_MOTHER) / E_V
    gV = E_V / np.sqrt(s)

    if fidelity == "legacy":
        cx = pstar * sr * np.cos(phr)
        cy = pstar * sr * np.sin(phr)
        cz = pstar * u
        tV = rng.uniform(0.0, np.pi, n)
        pV = rng.uniform(0.0, 2 * np.pi, n)
        nx, ny, nz = np.sin(tV) * np.cos(pV), np.sin(tV) * np.sin(pV), np.cos(tV)
    else:
        nx, ny, nz = pydecay.isotropic_dirs(rng, n)
        st = np.sqrt(np.clip(1.0 - nz * nz, 1e-30, None))
        e1x, e1y, e1z = -ny / st, nx / st, np.zeros(n)
        e2x = ny * e1z - nz * e1y
        e2y = nz * e1x - nx * e1z
        e2z = nx * e1y - ny * e1x
        cx = pstar * (sr * (np.cos(phr) * e1x + np.sin(phr) * e2x) + u * nx)
        cy = pstar * (sr * (np.cos(phr) * e1y + np.sin(phr) * e2y) + u * ny)
        cz = pstar * (sr * (np.cos(phr) * e1z + np.sin(phr) * e2z) + u * nz)

    hits = 0
    for sgn in (1.0, -1.0):
        mx, my, mz, mE = pydecay.boost(sgn * cx, sgn * cy, sgn * cz, Ec,
                                       bV * nx, bV * ny, bV * nz, gV)
        lx, ly, lz, _ = pydecay.boost(mx, my, mz, mE, px / E, py / E, pz / E,
                                      E / PI0_MOTHER)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = SIDE_D * ly / lx
            zc = SIDE_D * lz / lx
        hits += int(((lx > 0) & (np.abs(y) <= SIDE_HALF)
                     & (np.abs(zc) <= SIDE_HALF)).sum())
    return hits


def test_dalitz(quick: bool = False, verbose: bool = True) -> List[ParityRow]:
    """lanl_decayPion_12bar.cc vs pydecay (legacy + corrected fidelity)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    binary = compile_cpp(paths.legacy_dir() / "lanl_decayPion_12bar.cc",
                        paths.CACHE_DIR / "bin" / "lanl_decayPion_12bar")
    cfg = _cfg(quick)
    n = 400_000 if quick else N_PI0
    txt, P = _pi0_parents(cfg, n)
    rng = np.random.default_rng(cfg.engine.seed)

    rows = []
    masses = PI0_MASSES[:1] if quick else PI0_MASSES
    for m in masses:
        efftxt = SCRATCH / f"da_eff_{m:g}.txt"
        with open(SCRATCH / f"da_{m:g}.log", "w") as log:
            subprocess.run([str(binary), str(SCRATCH / f"da_{m:g}.root"), str(m),
                            str(efftxt), str(txt), str(SIDE_D), "10", "10",
                            str(PI0_MOTHER), "12345", "dalitz", "0", "3",
                            "rectangular"], check=True, stdout=log, stderr=log)
        row = np.atleast_1d(np.loadtxt(efftxt))
        c_eff = float(row[1])
        c_hits = float(row[2]) if row.size > 2 else c_eff * 2 * n
        c_err = _eff_err(c_hits, 2 * n)

        h_leg = _side_plane_hits(P, m, "legacy", rng)
        h_cor = _side_plane_hits(P, m, "corrected", rng)
        l_eff, l_err = h_leg / (2 * n), _eff_err(h_leg, 2 * n)
        k_eff = h_cor / (2 * n)

        z = (c_eff - l_eff) / np.hypot(c_err, l_err)
        rows.append(ParityRow("B:dalitz", m, c_eff, l_eff, float(z),
                              c_eff / l_eff if l_eff else float("inf"),
                              extra=k_eff))
        if verbose:
            print(f"  B m={m:6.3g}  cpp={c_eff:.4e}  py(legacy)={l_eff:.4e}  "
                  f"py(corr)={k_eff:.4e}  z={z:+.1f}")
    return rows


# ---------------------------------------------------------------- Test C ----
def test_fidelity_delta(quick: bool = False, verbose: bool = True) -> List[ParityRow]:
    """corrected/legacy Dalitz acceptance ratio in an off-axis geometry."""
    det = DetectorConfig(type="bar_array", distance_m=1000.0, bar_columns=4,
                         bar_rows=4, bar_layers=3, bar_size_m=0.05,
                         offaxis_mrad=4.0)
    rows = []
    grid = np.array(PI0_MASSES[:1] if quick else PI0_MASSES)
    for fid in ("legacy", "corrected"):
        cfg = Config(detector=det, engine=EngineConfig(fidelity=fid, quick=quick))
        scan_m, ageo, _ = pydecay.compute_acceptance(cfg, "pi0", grid,
                                                     verbose=False)
        rows.append((fid, dict(zip(scan_m, ageo))))
    out = []
    for m in grid:
        leg = rows[0][1].get(m, float("nan"))
        cor = rows[1][1].get(m, float("nan"))
        out.append(ParityRow("C:fidelity", float(m), leg, cor, 0.0,
                             cor / leg if leg else float("inf")))
        if verbose:
            print(f"  C m={m:6.3g}  legacy={leg:.4e}  corrected={cor:.4e}  "
                  f"corr/leg={cor/max(leg,1e-30):.4f}")
    return out


def run_all(quick: bool = False) -> List[ParityRow]:
    print("=== Test A: 2-body vs decayVectorMeson.cc ===")
    rows = test_2body(quick)
    print("=== Test B: Dalitz vs lanl_decayPion_12bar.cc ===")
    rows += test_dalitz(quick)
    print("=== Test C: corrected/legacy fidelity delta (off-axis) ===")
    rows += test_fidelity_delta(quick)
    return rows


if __name__ == "__main__":
    # Development entrypoint (not part of the user CLI):
    #   python -m mcpsim.validation.parity [--quick]
    import sys

    rows = run_all(quick="--quick" in sys.argv)
    bad = [r for r in rows if r.test.startswith(("A", "B")) and abs(r.z) > 3.0]
    for r in bad:
        print(f"PARITY FAIL {r.test} m={r.m_chi:g}: z={r.z:+.1f}")
    sys.exit(1 if bad else 0)
