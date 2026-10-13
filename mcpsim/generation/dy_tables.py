"""Built-in per-material Drell-Yan cross-section tables (mcpsim/data/drell_yan).

sigma_DY(m_chi) per target nucleon at eps = 1: LO q qbar -> gamma* -> chi chibar
(q = u d s c, MadGraph's `generate q q~ > chi chi~ / z`) for a proton beam on a
nucleus (A, Z), at the exact fixed-target sqrt(s), mu_F = mu_R = M(chi chibar),
M(chi chibar) >= 2 GeV and alpha = 1/127.9. These are the archived scans'
settings except for the parton densities (nCTEQ15, arXiv:1509.00792):

  beam    nCTEQ15_1_1, the free proton of the nCTEQ15 fit;
  target  the per-nucleon nucleus PDF (Z f^{p/A} + (A-Z) f^{n/A}) / A, which
          is nCTEQ15FullNuc_A_Z where that set exists. The bound-neutron PDF
          follows from isospin (u <-> d); the bound-proton PDF f^{p/A} depends
          on A only, and between fitted nuclei (molybdenum, A = 96) it is
          interpolated linearly in ln A (between A = 84 and 108).

The archived scans used the iron bound-proton set nCTEQ15_56_26 for BOTH beams
(stock MadGraph takes one lhaid): nuclear effects on the beam proton and no
target neutrons. fidelity = legacy keeps them; see docs/physics.md.

Each table also carries acc(m): the LO per-chi acceptance (pair at zero pT,
as in MadGraph LO events) of the Drell-Yan preset's detector at that beam
energy, computed with the same PDFs, which shape the pair rapidity. The header
names that detector (acc_detector, a geometry.face_signature); a run with
another detector is warned.

Regenerate (needs `pip install parton` and the LHAPDF grids of the sets in
NCTEQ15_SETS that bracket the materials, from
https://lhapdfsets.web.cern.ch/current/<set>.tar.gz):

    python -m mcpsim.generation.dy_tables --pdfsets DIR
"""

from __future__ import annotations

import argparse
import datetime
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .. import geometry, paths
from ..config import DetectorConfig
from ..physics import constants as C
from ..physics.dy_lo import PDF, acceptance_per_chi, fixed_target_s, sigma_hadronic_pb

# Bound-proton sets of the nCTEQ15 fit by mass number (they depend on A only:
# e.g. nCTEQ15_40_18 and nCTEQ15_40_20 are identical). A = 1 is the free proton.
NCTEQ15_SETS: Dict[int, str] = {
    1: "nCTEQ15_1_1", 3: "nCTEQ15_3_2", 4: "nCTEQ15_4_2", 6: "nCTEQ15_6_3",
    7: "nCTEQ15_7_3", 9: "nCTEQ15_9_4", 12: "nCTEQ15_12_6", 14: "nCTEQ15_14_7",
    20: "nCTEQ15_20_10", 27: "nCTEQ15_27_13", 40: "nCTEQ15_40_20",
    56: "nCTEQ15_56_26", 64: "nCTEQ15_64_32", 84: "nCTEQ15_84_42",
    108: "nCTEQ15_108_54", 119: "nCTEQ15_119_59", 131: "nCTEQ15_131_54",
    184: "nCTEQ15_184_74", 197: "nCTEQ15_197_79", 207: "nCTEQ15_207_103",
    208: "nCTEQ15_208_82",
}
BEAM_PDF_SET = NCTEQ15_SETS[1]

# The archived scans' settings, kept (see generation/madgraph.py).
PAIR_MASS_MIN_GEV = 2.0
ALPHA = C.ALPHA_EM_DY

_ISOSPIN = {1: 2, 2: 1, -1: -2, -2: -1}   # bound proton -> bound neutron


def bound_proton_weights(A: int) -> List[Tuple[str, float]]:
    """[(nCTEQ15 set, weight)] giving f^{p/A}: the set itself for a fitted
    nucleus, else linear in ln A between the two fitted neighbours."""
    if A in NCTEQ15_SETS:
        return [(NCTEQ15_SETS[A], 1.0)]
    nodes = sorted(NCTEQ15_SETS)
    if not nodes[0] < A < nodes[-1]:
        raise ValueError(f"A = {A} is outside the nCTEQ15 nuclei ({nodes[0]}-{nodes[-1]}).")
    hi = next(n for n in nodes if n > A)
    lo = max(n for n in nodes if n < A)
    w_hi = math.log(A / lo) / math.log(hi / lo)
    return [(NCTEQ15_SETS[lo], 1.0 - w_hi), (NCTEQ15_SETS[hi], w_hi)]


@lru_cache(maxsize=None)
def _grid_pdf(name: str, pdfdir: str) -> PDF:
    """Number density f(pid, x, Q) of an LHAPDF set's central member, with Q
    frozen at the set's QMin below it (LHAPDF's default extrapolation)."""
    try:
        import parton
    except ImportError as exc:
        raise ImportError("building Drell-Yan tables needs `pip install parton`.") from exc
    if not (Path(pdfdir) / name).is_dir():
        raise FileNotFoundError(
            f"LHAPDF set {name} not found in {pdfdir}; download it from "
            f"https://lhapdfsets.web.cern.ch/current/{name}.tar.gz")
    pdf = parton.mkPDF(name, 0, pdfdir=pdfdir)
    q_min = float(pdf.pdfset.info["QMin"])

    def f(pid: int, x: np.ndarray, q: np.ndarray) -> np.ndarray:
        q = np.maximum(q, q_min)
        return np.asarray(pdf.xfxQ(pid, x, q, grid=False), dtype=float).reshape(-1) / x
    return f


def nucleon_pdf(A: int, Z: int, pdfdir: str) -> PDF:
    """Per-nucleon PDF of the nucleus (A, Z): (Z f^{p/A} + (A-Z) f^{n/A}) / A."""
    parts = [(_grid_pdf(name, pdfdir), w) for name, w in bound_proton_weights(A)]
    z = Z / A

    def bound_proton(pid: int, x: np.ndarray, q: np.ndarray) -> np.ndarray:
        return sum(w * f(pid, x, q) for f, w in parts)

    def f(pid: int, x: np.ndarray, q: np.ndarray) -> np.ndarray:
        if z == 1.0:
            return bound_proton(pid, x, q)
        return z * bound_proton(pid, x, q) + (1.0 - z) * bound_proton(_ISOSPIN.get(pid, pid), x, q)
    return f


def mass_number(material: C.TargetMaterial) -> int:
    """Nucleons per nucleus of the (isotope-averaged) material."""
    return int(round(material.A))


def detector_acceptance(m_chi: float, energy_gev: float, beam: PDF, target: PDF,
                        det: DetectorConfig) -> float:
    """LO per-chi acceptance of `det` (exact cone for an on-axis disc, the
    azimuth-averaged face kernel otherwise)."""
    if det.type == "cylindrical" and not det.offaxis_mrad:
        theta = geometry.cylindrical_theta_cut(det.radius_m, det.distance_m)
        return acceptance_per_chi(m_chi, energy_gev, beam, target, theta,
                                  m_pair_min=PAIR_MASS_MIN_GEV)
    return acceptance_per_chi(m_chi, energy_gev, beam, target,
                              hit_fraction=lambda r: geometry.hit_fraction(det, r),
                              distance_m=det.distance_m, m_pair_min=PAIR_MASS_MIN_GEV)


def compute_table(material: str, energy_gev: float, masses: Iterable[float], pdfdir: str,
                  det: DetectorConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """(masses, columns [sigma_pb, acc], header) for the masses with sigma > 0;
    acc is the LO per-chi acceptance of the detector `det`."""
    mat = C.TARGET_MATERIALS[material]
    A, Z = mass_number(mat), mat.Z
    s = fixed_target_s(energy_gev)
    m = np.asarray(sorted(masses), dtype=float)
    m = m[2.0 * m < math.sqrt(s)]
    beam, target = nucleon_pdf(1, 1, pdfdir), nucleon_pdf(A, Z, pdfdir)
    sigma = np.array([sigma_hadronic_pb(mi, s, beam, target, m_pair_min=PAIR_MASS_MIN_GEV,
                                        alpha=ALPHA) for mi in m])
    if not np.all(np.isfinite(sigma)):
        raise RuntimeError(f"non-finite Drell-Yan cross section for {material} at {energy_gev} GeV.")
    # Right at the kinematic limit the PDF grids' interpolation undershoots
    # to |sigma| ~ 1e-18 pb of either sign; the table ends where sigma > 0.
    m, sigma = m[sigma > 0], sigma[sigma > 0]
    acc = np.array([detector_acceptance(mi, energy_gev, beam, target, det) for mi in m])
    from .. import __version__
    sets = " + ".join(f"{w:.4f} x {n}" for n, w in bound_proton_weights(A))
    header = {
        "material": material, "A": str(A), "Z": str(Z),
        "beam_energy_gev": f"{energy_gev:g}", "sqrt_s_gev": f"{math.sqrt(s):.4f}",
        "process": "LO q q~ > chi chi~ / z (q = u d s c), mu_F = mu_R = M(chi chi~), "
                   f"M(chi chi~) >= {PAIR_MASS_MIN_GEV:g} GeV, alpha = 1/{1 / ALPHA:g}",
        "beam_pdf": f"{BEAM_PDF_SET} (free proton)",
        "target_pdf": f"({Z} f^p/A + {A - Z} f^n/A) / {A}, f^p/A = {sets}",
        "acc": "LO per-chi acceptance of acc_detector (pair at zero pT), with these PDFs",
        "acc_detector": geometry.face_signature(det),
        "generator": f"mcpsim.generation.dy_tables (mcpsim {__version__}), "
                     f"{datetime.date.today().isoformat()}",
    }
    return m, np.column_stack([sigma, acc]), header


def write_table(path: Path, masses: np.ndarray, columns: np.ndarray, header: Dict[str, str]) -> Path:
    """Write compute_table's output: header lines, then m_chi, sigma, acc rows."""
    lines = ["# mcpsim Drell-Yan table: sigma(p N -> chi chi~ X) per target nucleon at eps = 1"]
    lines += [f"# {k}: {v}" for k, v in header.items()]
    lines.append("# columns: m_chi [GeV], sigma [pb], acc")
    lines += [f"{mi:.6e} {sigma:.6e} {acc:.6e}" for mi, (sigma, acc) in zip(masses, columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def table_name(material: str, energy_gev: float) -> str:
    return f"dy_sigma_{material}_{energy_gev:g}GeV.txt"


def preset_dy_inputs() -> Tuple[Dict[float, DetectorConfig], np.ndarray]:
    """{beam energy: detector} of the presets with a Drell-Yan channel, and the
    union of their DY acceptance-file masses (so preset runs read the tables
    at their nodes)."""
    from ..config import list_presets, load_preset
    detectors: Dict[float, DetectorConfig] = {}
    masses = []
    for name in list_presets():
        cfg = load_preset(name)
        if "drell_yan" in cfg.channels and cfg.data.dy_ageo:
            detectors[cfg.beam.energy_gev] = cfg.detector
            masses.append(np.atleast_2d(np.loadtxt(paths.resolve_data(cfg.data.dy_ageo)))[:, 0])
    return detectors, np.unique(np.concatenate(masses))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdfsets", required=True, help="directory holding the LHAPDF sets")
    ap.add_argument("--out", default=str(paths.DY_TABLES_DIR))
    ap.add_argument("--material", action="append", choices=sorted(C.TARGET_MATERIALS),
                    help="default: every material in physics/constants.TARGET_MATERIALS")
    ap.add_argument("--energy", action="append", type=float,
                    help="beam energy [GeV]; default: the Drell-Yan presets' energies")
    ap.add_argument("--detector-from", metavar="PRESET",
                    help="take the acceptance detector from this preset; default: "
                         "the Drell-Yan preset at that energy")
    args = ap.parse_args(argv)
    detectors, masses = preset_dy_inputs()
    for energy in args.energy or sorted(detectors):
        if args.detector_from:
            from ..config import load_preset
            det = load_preset(args.detector_from).detector
        elif energy in detectors:
            det = detectors[energy]
        else:
            ap.error(f"no Drell-Yan preset at {energy:g} GeV to take the detector "
                     f"from; give --detector-from PRESET")
        for material in args.material or sorted(C.TARGET_MATERIALS):
            m, cols, header = compute_table(material, energy, masses, args.pdfsets, det)
            path = write_table(Path(args.out) / table_name(material, energy), m, cols, header)
            print(f"wrote {path} ({m.size} masses)")


if __name__ == "__main__":
    main()
