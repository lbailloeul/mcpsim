"""Drell-Yan (qq-bar -> gamma* -> chi chi-bar) yield from precomputed inputs.

The per-nucleon cross section sigma(m) [pb] and the geometric acceptance
ageo(m) come either from the paper's MadGraph scans (a sigma column
POSITIONALLY row-aligned with a (mass, acceptance) file) or from mcpsim's
per-material tables (mass-keyed, with the LO acceptance of the preset's
detector; see load_table and generation/dy_tables.py). Factored from
dyProduction (DrawProductionDarkQuest.py).

Normalization: the per-nucleon Drell-Yan cross section over the per-nucleon
share of the target's interaction cross section, times the POT,

    N(m) = N_POT * sigma_DY(m) / sigma_inel * ageo(m),

where sigma_inel is the per-nucleon share of the target's interaction cross
section, sigma_inel(p-target)/A (sigma_inel_per_nucleon_mb). The published
pipeline hard-coded 13 mb (SIGMA_PN_INELASTIC_MB) = the iron value for every
target; that constant is kept for fidelity = legacy. The legacy code wrote this
as ``sigma*1e-12 / 13e-3`` with mismatched unit labels; the expression below
is numerically identical (regression-tested) with honest units.

Known data fragility (see the dev-level DY audit and docs/data.md): the cross
file has NO mass column, so pairing with the ageo file is positional. If the
row counts ever differ, dy_yield truncates to the shorter length for legacy
parity and the mismatch is surfaced as a warning upstream. (The shipped
DarkQuest and SHiP pairs are verified row-aligned.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import MB_TO_PB, N_AVOGADRO, SIGMA_PN_INELASTIC_MB


def sigma_inel_per_nucleon_mb(density_g_cm3: float, interaction_length_cm: float) -> float:
    """sigma_inel(p-target)/A [mb] = 1/(rho * lambda * N_A).

    Dividing a per-nucleon sigma_DY by this is the same as multiplying by the
    nucleons per cm^2 in one interaction length (the brem luminosity times A,
    paper eq. 4.14). Iron gives 12.6 mb, molybdenum 10.7, graphite 19.4."""
    return 1.0 / (density_g_cm3 * interaction_length_cm * N_AVOGADRO) * 1e27


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


@dataclass
class DYTable:
    """A mass-keyed per-material table (generation/dy_tables.py): the
    per-nucleon cross section and the LO per-chi acceptance of the detector
    named in the header (acc_detector, a geometry.face_signature)."""
    masses: np.ndarray
    sigma_pb: np.ndarray
    acc: Optional[np.ndarray]
    header: Dict[str, str]

    @property
    def acc_detector(self) -> Optional[str]:
        return self.header.get("acc_detector")

    def sigma_at(self, masses: np.ndarray) -> np.ndarray:
        return sigma_from_table(self.masses, self.sigma_pb, masses)


def load_table(path: str | Path) -> DYTable:
    """Read a table: '# key: value' header lines, then rows of
    m_chi [GeV], sigma [pb] and (optionally) the acceptance."""
    header: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].partition(":")
            if sep:
                header[key.strip()] = value.strip()
    data = np.atleast_2d(np.loadtxt(path))
    return DYTable(data[:, 0], data[:, 1], data[:, 2] if data.shape[1] > 2 else None, header)


def sigma_from_table(table_masses: np.ndarray, table_sigma: np.ndarray,
                     masses: np.ndarray) -> np.ndarray:
    """Table cross sections at `masses`: log-log interpolation, flat below the
    first mass (the pair-mass cut makes sigma mass-independent there), zero
    beyond the last (the kinematic reach of the table)."""
    pos = table_sigma > 0
    tm, ts = np.log(table_masses[pos]), np.log(table_sigma[pos])
    m = np.asarray(masses, dtype=float)
    sigma = np.exp(np.interp(np.log(m), tm, ts))
    sigma[m > table_masses[pos][-1] * (1 + 1e-9)] = 0.0
    return sigma


def find_table(directory: str | Path, energy_gev: float, material: str) -> Optional[Path]:
    """The table dy_sigma_<material>_<E>GeV.txt in `directory` whose beam energy
    is within 2% of energy_gev, or None."""
    for path in sorted(Path(directory).glob(f"dy_sigma_{material}_*GeV.txt")):
        try:
            e = float(path.stem.rsplit("_", 1)[1][:-3])
        except ValueError:
            continue
        if abs(energy_gev - e) <= 0.02 * e:
            return path
    return None


def dy_yield_table(table: DYTable, n_pot: float, sigma_inelastic_mb: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """DY yield (N_chi/eps^2) at the table's masses, from its per-nucleon
    cross section and its LO acceptance (no positional pairing, no ageo file)."""
    if table.acc is None:
        raise ValueError("this Drell-Yan table has no acceptance column")
    yields = n_pot * table.sigma_pb / (sigma_inelastic_mb * MB_TO_PB) * table.acc
    return table.masses, yields


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
