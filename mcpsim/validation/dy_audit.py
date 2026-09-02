"""Drell-Yan input audit (tiers 1-2 of the DY validation plan).

Tier 1 — normalization: prove the rewritten yield formula
    N = N_POT * sigma_pb / (SIGMA_PN_INELASTIC_MB * MB_TO_PB) * ageo
is numerically identical to the legacy expression
    N = N_POT * (sigma_pb * 1e-12) / 13e-3 * ageo
on the preset's actual input files (the legacy labels — "pb -> cm^2" divided
by a "pb-scale total" — were dimensionally muddled but the number was right).

Tier 2 — provenance/consistency of the input files:
  * row-count pairing (the cross file has no mass column; the SHiP pair is
    257 vs 258 rows — every row past the orphan may be mis-paired),
  * naming vs preset physics (nCTEQ15_iron files on the molybdenum SHiP
    preset; nCTEQ15 vs nCTEQ spelling),
  * mass-grid coverage vs the preset grid.

Tier 3 (MadGraph rerun with the in-repo UFO model) is documented in
docs/physics.md; it needs mg5_aMC + LHAPDF and is not run here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .. import paths
from ..config import Config
from ..physics.constants import MB_TO_PB, SIGMA_PN_INELASTIC_MB
from ..physics.drellyan import dy_yield


@dataclass
class DYAuditReport:
    preset: str
    identity_ok: bool
    identity_max_rel: float
    findings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.identity_ok


def audit(cfg: Config, verbose: bool = True) -> DYAuditReport:
    """Run tiers 1-2 against a config's DY inputs."""
    rep = DYAuditReport(preset=cfg.name, identity_ok=True, identity_max_rel=0.0)

    if not (cfg.data.dy_cross and cfg.data.dy_ageo):
        rep.findings.append("no Drell-Yan inputs configured for this preset.")
        rep.identity_ok = True
        _emit(rep, verbose)
        return rep

    cross_path = paths.resolve_data(cfg.data.dy_cross)
    ageo_path = paths.resolve_data(cfg.data.dy_ageo)

    # ---- tier 1: normalization identity ----------------------------------
    mass, new_y = dy_yield(str(cross_path), str(ageo_path), cfg.beam.n_pot)
    cross = np.atleast_1d(np.loadtxt(cross_path))
    sigma_pb = cross if cross.ndim == 1 else cross[:, 0]
    ageo_data = np.atleast_2d(np.loadtxt(ageo_path))
    n = min(sigma_pb.size, ageo_data.shape[0])
    legacy_y = cfg.beam.n_pot * (sigma_pb[:n] * 1e-12) / 13e-3 * ageo_data[:n, 1]
    denom = np.where(np.abs(legacy_y) > 0, np.abs(legacy_y), 1.0)
    rel = float(np.max(np.abs(new_y - legacy_y) / denom))
    rep.identity_max_rel = rel
    rep.identity_ok = rel < 1e-12
    # Sanity: the named constants reproduce the legacy magic number.
    assert abs(1.0 / (SIGMA_PN_INELASTIC_MB * MB_TO_PB) - 1e-12 / 13e-3) < 1e-25

    # ---- tier 2: provenance / consistency --------------------------------
    if sigma_pb.size != ageo_data.shape[0]:
        d = ageo_data.shape[0] - sigma_pb.size
        rep.findings.append(
            f"ROW MISMATCH: {cross_path.name} has {sigma_pb.size} rows, "
            f"{ageo_path.name} has {ageo_data.shape[0]} ({d:+d}). Pairing is "
            f"positional (no mass column in the cross file); every pair after "
            f"the orphan row may be shifted. The orphan's location cannot be "
            f"recovered from the files alone — re-export both from the same "
            f"MadGraph scan to fix."
        )

    cross_name, ageo_name = str(cfg.data.dy_cross), str(cfg.data.dy_ageo)
    if "iron" in (cross_name + ageo_name) and cfg.target.name != "iron":
        rep.findings.append(
            f"NAMING vs PHYSICS: DY files are tagged '_iron' "
            f"({cross_name}, {ageo_name}) but the preset target is "
            f"{cfg.target.name}. Either the files genuinely used an iron "
            f"nuclear PDF (a documented approximation) or they are mislabeled "
            f"— the generating MadGraph cards are not preserved, so this "
            f"cannot be resolved from the data; record the decision in "
            f"docs/data.md."
        )
    def _pdf_tag(name: str):
        return "nCTEQ15" if "nCTEQ15" in name else ("nCTEQ" if "nCTEQ" in name else None)

    t_cross, t_ageo = _pdf_tag(cross_name), _pdf_tag(ageo_name)
    if t_cross and t_ageo and t_cross != t_ageo:
        rep.findings.append(
            f"NAMING: inconsistent PDF-set tags across the pair "
            f"({cross_name}: '{t_cross}' vs {ageo_name}: '{t_ageo}'). Assumed "
            f"to be the same set; the mg5 lhaid is not recorded anywhere."
        )

    grid = np.atleast_1d(np.loadtxt(paths.resolve_data(cfg.data.mass_grid)))
    lo, hi = mass.min(), mass.max()
    n_out = int(np.sum((grid < lo) | (grid > hi)))
    if n_out:
        rep.findings.append(
            f"coverage: DY tabulated for m in [{lo:g}, {hi:g}] GeV; "
            f"{n_out}/{grid.size} grid masses fall outside and are zero-filled."
        )

    _emit(rep, verbose)
    return rep


def _emit(rep: DYAuditReport, verbose: bool) -> None:
    if not verbose:
        return
    status = "PASS" if rep.identity_ok else "FAIL"
    print(f"[dy-audit] {rep.preset}: normalization identity {status} "
          f"(max rel diff {rep.identity_max_rel:.2e})")
    for f in rep.findings:
        print(f"[dy-audit]   - {f}")


if __name__ == "__main__":
    # Development entrypoint (not part of the user CLI):
    #   python -m mcpsim.validation.dy_audit [preset ...]
    import sys

    from ..config import load_preset

    presets = [a for a in sys.argv[1:] if not a.startswith("-")] or ["darkquest", "ship"]
    ok = True
    for name in presets:
        ok = audit(load_preset(name)).ok and ok
    sys.exit(0 if ok else 1)
