"""Drell-Yan (qq-bar -> gamma* -> chi chi-bar) generation via MadGraph.

Generates, per mCP mass, the cross section sigma(m) and the geometry-specific
acceptance(m), writing them in the drop-in format fast mode consumes
(dy_cross_*.txt single-column pb, ageo_*.txt two-column mass+acceptance).

Millicharged particles are not a built-in MG5 model, so the UFO model directory
must be supplied via the config ('dy_model' / data) or the MCPSIM_MCP_MODEL
environment variable. Requires mg5_aMC on PATH (and LHAPDF for nCTEQ15 PDFs).

The LHE parser is deliberately standalone so it can be unit-tested without
MadGraph.
"""

from __future__ import annotations

import gzip
import math
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .. import geometry, paths
from ..config import Config
from . import common

# PDG id of the millicharged fermion: 31 in BOTH the original Minimal_MCP UFO
# (Zenodo 10.5281/zenodo.18330380) and the in-repo models/mcp_ufo, matching
# the PYTHIA samples' mcp id (momentum.config: '31:new'). Override with
# MCPSIM_MCP_PDG for a model that uses a different id.
MCP_PDG_DEFAULT = int(os.environ.get("MCPSIM_MCP_PDG", "31"))

PROC_CARD_TEMPLATE = """\
import model {model}
define q = u d s c u~ d~ s~ c~
generate q q~ > chi chi~ / z
output {outdir}
"""

RUN_CARD_SETTINGS = {
    "lpp1": "1", "lpp2": "1",
    "pdlabel": "lhapdf", "lhaid": "30000",   # nCTEQ15-ish; override per setup
    "use_syst": "False",
}


def generate_drell_yan(cfg: Config, masses: np.ndarray, work_dir: Path) -> Tuple[Path, Path]:
    """Run MadGraph over the mass grid; return (dy_cross_file, dy_ageo_file)."""
    common.require_command("mg5_aMC", "Install MadGraph5_aMC@NLO and add mg5_aMC to PATH.")
    model = _resolve_model(cfg)
    work_dir.mkdir(parents=True, exist_ok=True)

    cross_rows: List[float] = []
    ageo_rows: List[Tuple[float, float]] = []
    for mchi in np.atleast_1d(masses):
        proc_dir = work_dir / f"dy_m{mchi:.6g}"
        sigma_pb, lhe = _run_single_mass(cfg, model, float(mchi), proc_dir)
        n_tot, n_acc = parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg)
        acc = (n_acc / n_tot) if n_tot else 0.0
        cross_rows.append(sigma_pb)
        ageo_rows.append((float(mchi), acc))

    tag = re.sub(r"\W+", "_", cfg.name.lower())
    dy_cross = work_dir / f"dy_cross_{tag}.txt"
    dy_ageo = work_dir / f"ageo_dy_{tag}.txt"
    np.savetxt(dy_cross, np.array(cross_rows), fmt="%.6g")
    np.savetxt(dy_ageo, np.array(ageo_rows), fmt="%.8g")
    return dy_cross, dy_ageo


def _resolve_model(cfg: Config) -> str:
    """UFO model directory: MCPSIM_MCP_MODEL env var, else the in-repo
    millicharged-fermion recreation (models/mcp_ufo). The original model the
    archived DY scans were generated with is `Minimal_MCP` on the paper's
    Zenodo record; see docs/data.md."""
    model = os.environ.get("MCPSIM_MCP_MODEL")
    if model:
        return model
    bundled = paths.PROJECT_ROOT / "models" / "mcp_ufo"
    if (bundled / "__init__.py").exists():
        return str(bundled)
    raise common.ToolchainError(
        "No millicharged-particle MG5 model configured. Set MCPSIM_MCP_MODEL "
        "to your UFO model directory (the in-repo default models/mcp_ufo is "
        "only present in a source checkout)."
    )


def _run_single_mass(cfg: Config, model: str, mchi: float, proc_dir: Path) -> Tuple[float, Path]:
    """Write cards, launch mg5 for one mass, return (sigma_pb, lhe_path)."""
    proc_dir.parent.mkdir(parents=True, exist_ok=True)
    proc_card = proc_dir.parent / f"proc_{proc_dir.name}.mg5"
    proc_card.write_text(PROC_CARD_TEMPLATE.format(model=model, outdir=proc_dir))
    common.run(["mg5_aMC", str(proc_card)])

    _patch_run_card(proc_dir / "Cards" / "run_card.dat", cfg)
    _patch_param_card(proc_dir / "Cards" / "param_card.dat", mchi)
    common.run([str(proc_dir / "bin" / "generate_events"), "-f"])

    lhe = _find_lhe(proc_dir)
    sigma_pb = parse_cross_section(proc_dir)
    return sigma_pb, lhe


def _patch_run_card(path: Path, cfg: Config) -> None:
    if not path.exists():
        raise common.ToolchainError(f"run_card.dat not found at {path}.")
    text = path.read_text()
    settings = dict(RUN_CARD_SETTINGS)
    settings["ebeam1"] = f"{cfg.beam.energy_gev}"
    # Fixed target: second beam at rest (proton mass ~ 0 vs beam energy).
    settings["ebeam2"] = "0.938" if not cfg.beam.is_collider else f"{cfg.beam.energy_b_gev}"
    for key, val in settings.items():
        text = re.sub(rf"(?m)^(\s*)\S+(\s*=\s*{key}\b.*)$", rf"\g<1>{val}\g<2>", text)
    path.write_text(text)


def _patch_param_card(path: Path, mchi: float) -> None:
    if not path.exists():
        raise common.ToolchainError(f"param_card.dat not found at {path}.")
    text = path.read_text()
    # Set the mCP mass block entry for the MCP PDG id.
    text = re.sub(rf"(?m)^(\s*{MCP_PDG_DEFAULT}\s+)\S+(.*# *mchi.*)?$",
                  rf"\g<1>{mchi:.6e}\g<2>", text)
    path.write_text(text)


def _find_lhe(proc_dir: Path) -> Path:
    matches = sorted(proc_dir.glob("Events/**/unweighted_events.lhe*"))
    if not matches:
        raise common.ToolchainError(f"no LHE output found under {proc_dir}/Events.")
    return matches[-1]


def parse_cross_section(proc_dir: Path) -> float:
    """Read the integrated cross section (pb) from the MG results."""
    for results in proc_dir.glob("Events/**/*.txt"):
        m = re.search(r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)", results.read_text())
        if m:
            return float(m.group(1))
    raise common.ToolchainError(f"could not parse cross section under {proc_dir}/Events.")


def parse_lhe_acceptance(lhe_path: Path | str, mcp_pdg: int, cfg: Config) -> Tuple[int, int]:
    """Count events and geometry-accepted events from an LHE file.

    An event is accepted when at least one millicharged particle enters the
    detector aperture (cylindrical theta cut, or rectangular bar-array face).
    Returns (n_total, n_accepted). Works on plain or gzipped LHE.
    """
    det = cfg.detector
    opener = gzip.open if str(lhe_path).endswith(".gz") else open
    n_total = n_accepted = 0
    with opener(lhe_path, "rt") as fh:
        in_event = False
        accepted = False
        header_pending = False
        for line in fh:
            s = line.strip()
            if s.startswith("<event"):
                in_event, accepted, header_pending = True, False, True
                continue
            if s.startswith("</event>"):
                n_total += 1
                n_accepted += int(accepted)
                in_event = False
                continue
            if not in_event:
                continue
            if header_pending:           # first line after <event> is the header
                header_pending = False
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            try:
                pid = int(float(parts[0]))
                px, py, pz = float(parts[6]), float(parts[7]), float(parts[8])
            except ValueError:
                continue
            if abs(pid) == mcp_pdg and _in_acceptance(px, py, pz, det):
                accepted = True
    return n_total, n_accepted


def _in_acceptance(px: float, py: float, pz: float, det) -> bool:
    if pz <= 0:
        return False
    if det.type == "cylindrical":
        theta = math.atan2(math.hypot(px, py), pz)
        return theta < geometry.cylindrical_theta_cut(det.radius_m, det.distance_m)
    # bar-array rectangular face at z = distance.
    width, height = geometry.bar_array_dims(det)
    x_at = det.distance_m * px / pz
    y_at = det.distance_m * py / pz
    return abs(x_at) <= 0.5 * width and abs(y_at) <= 0.5 * height
