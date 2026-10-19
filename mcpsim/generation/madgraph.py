"""Drell-Yan (qq-bar -> gamma* -> chi chi-bar) generation via MadGraph.

Generates, per mCP mass, the cross section sigma(m) and the geometry-specific
acceptance(m), writing them in the drop-in format fast mode consumes
(dy_cross_*.txt single-column pb, ageo_*.txt two-column mass+acceptance).

Two execution modes, as for brem:
  * local (default)  -- one MadGraph process per mass, run in turn.
  * Condor (--condor) -- a batch bundle with one job per mass, each building
                        its MadGraph process in the worker's scratch space;
                        the last job to finish assembles the two files.
                        README_condor.txt in the bundle says how.

The millicharged fermion is not a built-in MG5 model: the in-repo UFO model
models/mcp_ufo is used unless the MCPSIM_MCP_MODEL environment variable names
another. Requires mg5_aMC on PATH, and LHAPDF with the nCTEQ15_56_26 grid.

The LHE parser is deliberately standalone so it can be unit-tested without
MadGraph.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__, geometry, paths
from ..config import Config, config_from_record
from ..physics.constants import DY_PAIR_MASS_MIN_GEV
from ..physics.dy_lo import fixed_target_s, fixed_target_y_cm
from . import common

# PDG id of the millicharged fermion: 31 in BOTH the original Minimal_MCP UFO
# (Zenodo 10.5281/zenodo.18330380) and the in-repo models/mcp_ufo, matching
# the PYTHIA samples' mcp id (momentum.config: '31:new'). Override with
# MCPSIM_MCP_PDG for a model that uses a different id.
MCP_PDG_DEFAULT = int(os.environ.get("MCPSIM_MCP_PDG", "31"))

# q q~ > chi chi~ with q = u d s c: separate quark and antiquark labels
# (MadGraph does not derive "q~" from a multiparticle), and the Z vetoed only
# when the model has one (the original Minimal_MCP couples chi to it; the
# in-repo QED-only models/mcp_ufo has no Z, and "/ z" would be an error).
# Both orderings: MadGraph adds the mirror process (antiquark from the beam)
# only when it is requested, and without it sigma is half the archived scans'
# (and the acceptance ~1.7x too high: a valence beam quark boosts the pair).
# "-f" replaces a process directory left by an earlier run instead of
# stopping at MadGraph's overwrite prompt.
PROC_CARD_TEMPLATE = """\
import model {model}
define q = u d s c
define q~ = u~ d~ s~ c~
generate q q~ > chi chi~{veto}
add process q~ q > chi chi~{veto}
output {outdir} -f
"""


def _model_has_z(model: str) -> bool:
    particles = Path(model) / "particles.py"
    return particles.exists() and re.search(r"pdg_code\s*=\s*23\b", particles.read_text()) is not None

# The settings that reproduce the archived DarkQuest and SHiP dy_cross files
# (an independent LO convolution matches them to 0.1-0.7%, see docs/data.md):
# the iron bound-proton nPDF nCTEQ15_56_26 for BOTH beams (stock MadGraph
# takes one lhaid), mu_F = mu_R = sqrt(s_hat) = M(chi chi~), a pair-mass cut
# M >= 2 GeV (below it vector mesons cover the region), alpha = 1/127.9 from
# the UFO model, and the exact fixed-target sqrt(s). A regenerated scan is
# used as it is, cross sections and the acceptance of the configured
# detector, in place of the built-in per-material table: the tables' free-
# proton beam on a nucleus target is beyond one lhaid (pipeline.
# regenerate_inputs; generation/dy_tables.py).
DY_PDF_SET, DY_PDF_LHAID, DY_PDF_MATERIAL = "nCTEQ15_56_26", 100600, "iron"
# MadGraph's default; ~3.5% statistics on a per-chi acceptance of 0.04.
DY_NEVENTS_DEFAULT = 10000

RUN_CARD_SETTINGS = {
    "lpp1": "1", "lpp2": "1",
    "pdlabel": "lhapdf", "lhaid": str(DY_PDF_LHAID),
    "dynamical_scale_choice": "4",                       # sqrt(s_hat)
    "mxx_min_pdg": f"{{{MCP_PDG_DEFAULT}: {DY_PAIR_MASS_MIN_GEV}}}",
    "use_syst": "False",
}


def generate_drell_yan(cfg: Config, masses: np.ndarray, work_dir: Path, *,
                       nevents: int = DY_NEVENTS_DEFAULT,
                       n_proc: Optional[int] = None) -> Tuple[Path, Path]:
    """Run MadGraph over the mass grid on n_proc cores (None: MadGraph's own
    setting); return (dy_cross_file, dy_ageo_file)."""
    common.require_command("mg5_aMC", "Install MadGraph5_aMC@NLO and add mg5_aMC to PATH.")
    work_dir.mkdir(parents=True, exist_ok=True)
    model = _local_model(_resolve_model(cfg), work_dir)

    masses = np.atleast_1d(masses).astype(float)
    sigmas: List[float] = []
    accs: List[float] = []
    for mchi in masses:
        sigma_pb, acc, _, _ = _run_mass(cfg, model, float(mchi), work_dir / f"dy_m{mchi:.6g}",
                                        nevents=nevents, cores=n_proc)
        sigmas.append(sigma_pb)
        accs.append(acc)
    return _write_scan(cfg, work_dir, masses, sigmas, accs)


def _run_mass(cfg: Config, model: str, mchi: float, proc_dir: Path, *,
              nevents: int, cores: Optional[int] = None) -> Tuple[float, float, int, int]:
    """MadGraph for one mass: (sigma_pb, per-chi acceptance, events, chi hits)."""
    sigma_pb, lhe = _run_single_mass(cfg, model, mchi, proc_dir, nevents=nevents, cores=cores)
    n_tot, n_hits = parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg, boost_y=lab_rapidity(cfg))
    acc = (n_hits / (2.0 * n_tot)) if n_tot else 0.0      # per chi
    return sigma_pb, acc, n_tot, n_hits


def _scan_paths(cfg: Config, work_dir: Path) -> Tuple[Path, Path]:
    return work_dir / f"dy_cross_{cfg.tag}.txt", work_dir / f"ageo_dy_{cfg.tag}.txt"


def _write_scan(cfg: Config, work_dir: Path, masses: Sequence[float],
                sigmas: Sequence[float], accs: Sequence[float]) -> Tuple[Path, Path]:
    """Write dy_cross_<tag>.txt and ageo_dy_<tag>.txt, each replaced whole:
    two batch jobs finishing together may both assemble the scan."""
    dy_cross, dy_ageo = _scan_paths(cfg, work_dir)
    for path, data, fmt in ((dy_cross, np.asarray(sigmas, dtype=float), "%.6g"),
                            (dy_ageo, np.column_stack([masses, accs]), "%.8g")):
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        np.savetxt(tmp, data, fmt=fmt)
        os.replace(tmp, path)
    return dy_cross, dy_ageo


def _local_model(model: str, work_dir: Path) -> str:
    """A copy of the UFO model in the work dir: MadGraph caches a pickle
    inside the model directory, so it never runs on the source tree's."""
    dest = work_dir / Path(model).name
    shutil.copytree(model, dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pkl"))
    return str(dest)


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


def _run_single_mass(cfg: Config, model: str, mchi: float, proc_dir: Path, *,
                     nevents: int = DY_NEVENTS_DEFAULT,
                     cores: Optional[int] = None) -> Tuple[float, Path]:
    """Write cards, launch mg5 for one mass, return (sigma_pb, lhe_path)."""
    proc_dir.parent.mkdir(parents=True, exist_ok=True)
    proc_card = proc_dir.parent / f"proc_{proc_dir.name}.mg5"
    proc_card.write_text(PROC_CARD_TEMPLATE.format(
        model=model, outdir=proc_dir, veto=" / z" if _model_has_z(model) else ""))
    # from the work dir: MadGraph leaves parser tables and debug links in its cwd
    common.run(["mg5_aMC", str(proc_card)], cwd=proc_dir.parent)

    _patch_run_card(proc_dir / "Cards" / "run_card.dat", cfg, nevents=nevents)
    _patch_param_card(proc_dir / "Cards" / "param_card.dat", mchi)
    if cores:
        _set_cores(proc_dir / "Cards" / "me5_configuration.txt", cores)
    common.run([str(proc_dir / "bin" / "generate_events"), "-f"], cwd=proc_dir)

    lhe = _find_lhe(proc_dir)
    sigma_pb = parse_cross_section(proc_dir)
    return sigma_pb, lhe


def lab_rapidity(cfg: Config) -> float:
    """Rapidity of the frame MadGraph's events are in, seen from the lab:
    the beam-nucleon centre of mass for a fixed target (see _patch_run_card),
    0 for a collider (events already in the lab)."""
    return 0.0 if cfg.beam.is_collider else fixed_target_y_cm(cfg.beam.energy_gev)


def _patch_run_card(path: Path, cfg: Config, *, nevents: int = DY_NEVENTS_DEFAULT) -> None:
    if not path.exists():
        raise common.ToolchainError(f"run_card.dat not found at {path}.")
    text = path.read_text()
    settings = dict(RUN_CARD_SETTINGS, nevents=str(int(nevents)))
    if cfg.beam.is_collider:
        settings.update(ebeam1=f"{cfg.beam.energy_gev}", ebeam2=f"{cfg.beam.energy_b_gev}")
    else:
        # Fixed target, generated in the beam-nucleon centre of mass with
        # massless beams of sqrt(s)/2 each (sqrt(s) = 15.065 GeV at 120 GeV)
        # and boosted to the lab by parse_lhe_acceptance. Not with the target
        # at rest (ebeam2 = 0.938): MadGraph then writes the events in a lab
        # frame rebuilt from a massive beam of energy E_parton/x in the
        # partonic frame (unwgt.f, write_leshouche), which is off when that
        # energy is close to m_p. At these masses (sqrt(s_hat) ~ 2 GeV) it
        # pushed large-x1 pairs forward by up to 0.19 in rapidity and raised
        # DarkQuest's acceptance by 3.4%. Massless beams make that frame exact.
        half = f"{math.sqrt(fixed_target_s(cfg.beam.energy_gev)) / 2:.6f}"
        settings.update(ebeam1=half, ebeam2=half, mass_ion1="0.0", mass_ion2="0.0")
    for key, val in settings.items():
        text, n = re.subn(rf"(?m)^(\s*)\S+(\s*=\s*{key}\b.*)$", rf"\g<1>{val}\g<2>", text)
        if n == 0:                       # a hidden parameter (mass_ion): add it
            text = text.rstrip("\n") + f"\n  {val} = {key}\n"
    path.write_text(text)


def _patch_param_card(path: Path, mchi: float) -> None:
    if not path.exists():
        raise common.ToolchainError(f"param_card.dat not found at {path}.")
    text = path.read_text()
    # The chi entry of Block MASS ("   31 1.000000e-01 # MCHI"), only inside
    # that block and whatever the comment's case; an unpatched card would run
    # every scan point at the model's default mass, so a miss is an error.
    block = re.search(r"(?ims)^block\s+mass\b.*?(?=^\s*(?:block|decay)\b|\Z)", text)
    if block is None:
        raise common.ToolchainError(f"no Block MASS in {path}.")
    body, n = re.subn(rf"(?m)^(\s*{MCP_PDG_DEFAULT}\s+)\S+", rf"\g<1>{mchi:.6e}", block.group(0))
    if n != 1:
        raise common.ToolchainError(
            f"could not set the mass of pdg {MCP_PDG_DEFAULT} in Block MASS of {path}.")
    path.write_text(text[:block.start()] + body + text[block.end():])


def _set_cores(path: Path, cores: int) -> None:
    """MadGraph's multicore mode on `cores` cores. Its default is every core
    of the machine, which on a batch worker is more than the job asked for."""
    if not path.exists():
        raise common.ToolchainError(f"me5_configuration.txt not found at {path}.")
    text = path.read_text()
    for key, val in (("run_mode", "2"), ("nb_core", str(int(cores)))):
        text, n = re.subn(rf"(?m)^#?\s*{key}\s*=.*$", f"{key} = {val}", text)
        if n == 0:
            raise common.ToolchainError(f"no {key} setting in {path}.")
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


def parse_lhe_acceptance(lhe_path: Path | str, mcp_pdg: int, cfg: Config,
                         boost_y: float = 0.0) -> Tuple[int, int]:
    """Count events and millicharged particles entering the detector face
    (the disc or the bar-array rectangle, at its off-axis offset) in an LHE
    file, after boosting the events along the beam by rapidity `boost_y` (the
    lab rapidity of their frame, lab_rapidity(cfg) for mcpsim's MadGraph runs).

    Returns (n_total, n_hits); n_hits counts chi and chi-bar separately, so
    n_hits / (2 n_total) is the per-chi acceptance — the convention of the
    archived ageo files and the meson channels (not the share of events with
    at least one hit, up to 2x larger). Works on plain or gzipped LHE.
    """
    det = cfg.detector
    ch, sh = math.cosh(boost_y), math.sinh(boost_y)
    opener = gzip.open if str(lhe_path).endswith(".gz") else open
    n_total = n_hits = 0
    with opener(lhe_path, "rt") as fh:
        in_event = False
        header_pending = False
        for line in fh:
            s = line.strip()
            if s.startswith("<event"):
                in_event, header_pending = True, True
                continue
            if s.startswith("</event>"):
                n_total += 1
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
                px, py, pz, e = (float(v) for v in parts[6:10])
            except ValueError:
                continue
            if abs(pid) == mcp_pdg and _in_acceptance(px, py, pz * ch + e * sh, det):
                n_hits += 1
    return n_total, n_hits


def _in_acceptance(px: float, py: float, pz: float, det) -> bool:
    if pz <= 0:
        return False
    # The face centre sits at the off-axis offset d along +x on the plane
    # z = distance, as in geometry.hit_fraction.
    d = float(geometry.offaxis_offset_m(det))
    if det.type == "cylindrical":
        # |crossing point - centre| < radius, kept in the cone-cut form so an
        # on-axis disc gives the same verdicts to the last bit.
        theta = math.atan2(math.hypot(px - d * pz / det.distance_m, py), pz)
        return theta < geometry.cylindrical_theta_cut(det.radius_m, det.distance_m)
    # bar-array rectangular face at z = distance.
    width, height = geometry.bar_array_dims(det)
    x_at = det.distance_m * px / pz - d
    y_at = det.distance_m * py / pz
    return abs(x_at) <= 0.5 * width and abs(y_at) <= 0.5 * height


# -- HTCondor batch: one job per mass ------------------------------------------
#
# The bundle carries everything a job needs: the run config (config.json, the
# Config as dataclasses.asdict), the mass list, a copy of the UFO model and
# bundle.json (model, events, cores). Each job runs dy_job.py, which imports
# the mcpsim that wrote the bundle when it was a source checkout, so the
# workers need no install of their own.

def prepare_condor_dy(cfg: Config, masses: np.ndarray, work_dir: Path, *,
                      n_proc: int = 4, nevents: int = DY_NEVENTS_DEFAULT,
                      submit: bool = True) -> Tuple[Path, Path]:
    """Emit (and optionally submit) an HTCondor bundle for the Drell-Yan scan.

    Submits via condor_submit when available; raises if submission was asked
    for without it, and with submit=False prints the command. Returns the
    (dy_cross, dy_ageo) paths the batch fills when its last job finishes."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "condor_logs").mkdir(exist_ok=True)
    (work_dir / "dy_out").mkdir(exist_ok=True)

    masses = np.atleast_1d(masses).astype(float)
    model = Path(_local_model(_resolve_model(cfg), work_dir))
    np.savetxt(work_dir / "dy_masses.txt", masses, fmt="%.10g")
    (work_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    (work_dir / "bundle.json").write_text(json.dumps(
        {"model": model.name, "nevents": int(nevents), "cores": int(n_proc),
         "mcpsim_version": __version__}, indent=1))

    # The mcpsim and MadGraph that wrote the bundle become the jobs' defaults.
    src = paths.PROJECT_ROOT if (paths.PROJECT_ROOT / "pyproject.toml").exists() else None
    mg5 = shutil.which("mg5_aMC")
    mg5_dir = str(Path(mg5).resolve().parent.parent) if mg5 else ""
    (work_dir / "dy_job.py").write_text(_DY_JOB_LAUNCHER.format(
        sys_path=f"sys.path.insert(0, {str(src)!r})" if src else
        "# (written by an installed mcpsim: the jobs import the installed one)"))
    wrapper = work_dir / "run_dy_job.sh"
    wrapper.write_text(_DY_CONDOR_WRAPPER.format(
        n_proc=n_proc, mg5_dir=mg5_dir, workdir=work_dir))
    wrapper.chmod(0o755)
    (work_dir / "dy.sub").write_text(_DY_CONDOR_SUBMIT.format(
        n_proc=n_proc, n_masses=masses.size))

    dy_cross, dy_ageo = _scan_paths(cfg, work_dir)
    (work_dir / "README_condor.txt").write_text(_DY_CONDOR_README.format(
        n_masses=masses.size, name=cfg.name, energy=cfg.beam.energy_gev, nevents=nevents,
        pdf_set=DY_PDF_SET, lhaid=DY_PDF_LHAID,
        workdir=work_dir, dy_cross=dy_cross, dy_ageo=dy_ageo,
        mg5_line=(f"The jobs use MG5_DIR = {mg5_dir}\n   (the one on PATH when "
                  f"this bundle was written), else mg5_aMC on PATH." if mg5_dir else
                  "No mg5_aMC was on PATH when this bundle was written: set\n"
                  "   MG5_DIR (see above) to your MadGraph directory."),
        python_step=(f"The jobs import mcpsim from the checkout that wrote\n"
                     f"   this bundle, {src},\n   so the LCG view's python3 "
                     f"(numpy, PyYAML) is enough. A venv also works:\n"
                     f"     python3 -m venv dy_venv && dy_venv/bin/pip install -e {src}"
                     if src else
                     f"Make a venv with mcpsim {__version__} (once):\n"
                     f"     python3 -m venv dy_venv && dy_venv/bin/pip install <your mcpsim source>\n"
                     f"   or set PYTHON_BIN to a Python that has it.")))

    if submit and shutil.which("condor_submit"):
        common.run(["condor_submit", "dy.sub"], cwd=work_dir)
        print(f"[mcpsim] submitted {masses.size} Condor jobs; the last to finish "
              f"writes {dy_cross.name} and {dy_ageo.name} in {work_dir}.")
    elif submit:
        raise common.ToolchainError(
            f"Condor bundle written to {work_dir} ({masses.size} jobs) but "
            f"'condor_submit' is not on PATH — nothing was submitted. Submit "
            f"from a Condor node with: cd {work_dir} && condor_submit dy.sub "
            f"(or use --no-submit to silence this)."
        )
    else:
        print(f"[mcpsim] Condor bundle (--no-submit) written to {work_dir} "
              f"({masses.size} jobs); see README_condor.txt there. Submit with:\n"
              f"  cd {work_dir} && condor_submit dy.sub")
    return dy_cross, dy_ageo


def _bundle(bundle: Path) -> Tuple[Config, dict, np.ndarray]:
    cfg = config_from_record(json.loads((bundle / "config.json").read_text()))
    meta = json.loads((bundle / "bundle.json").read_text())
    return cfg, meta, np.atleast_1d(np.loadtxt(bundle / "dy_masses.txt"))


def run_condor_job(bundle: Path, index: int, *, cores: Optional[int] = None,
                   nevents: Optional[int] = None) -> Path:
    """One batch job: MadGraph for mass `index` of the bundle, built in the
    worker's scratch space (_CONDOR_SCRATCH_DIR, else <bundle>/work). Writes
    dy_out/dy_<index>.txt and, if it is the last job, the scan files."""
    bundle = Path(bundle).resolve()
    cfg, meta, masses = _bundle(bundle)
    if not 0 <= index < masses.size:
        raise common.ToolchainError(
            f"job index {index} out of range: the bundle has {masses.size} masses.")
    mchi = float(masses[index])
    nevents = int(nevents or meta["nevents"])
    cores = int(cores or meta["cores"])

    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    job_dir = (Path(scratch) if scratch else bundle / "work") / f"dy_job_{index:04d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    # Its own model copy: MadGraph writes into the model it imports.
    model = _local_model(str(bundle / meta["model"]), job_dir)
    proc_dir = job_dir / f"dy_m{mchi:.6g}"
    sigma_pb, acc, n_tot, n_hits = _run_mass(cfg, model, mchi, proc_dir,
                                             nevents=nevents, cores=cores)

    out_dir = bundle / "dy_out"
    out_dir.mkdir(exist_ok=True)
    for banner in sorted(proc_dir.glob("Events/*/*_banner.txt"))[:1]:
        shutil.copyfile(banner, out_dir / f"banner_{index:04d}.txt")
    out = out_dir / f"dy_{index:04d}.txt"
    tmp = out.with_name(f".{out.name}.{os.getpid()}")
    tmp.write_text("# index mass_GeV sigma_pb acc_per_chi n_events n_chi_hits nevents_requested\n"
                   f"{index} {mchi:.10g} {sigma_pb:.6g} {acc:.8g} {n_tot} {n_hits} {nevents}\n")
    os.replace(tmp, out)
    print(f"[mcpsim] m_chi = {mchi:g} GeV: sigma = {sigma_pb:.6g} pb, "
          f"acceptance = {acc:.4g} ({n_hits} chi hits in {n_tot} events) -> {out}")

    try:
        files = collect_condor_dy(bundle)
    except common.ToolchainError:
        return out                       # other jobs still to finish
    print(f"[mcpsim] all {masses.size} jobs done: wrote {files[0]} and {files[1]}")
    return out


def collect_condor_dy(bundle: Path) -> Tuple[Path, Path]:
    """Assemble a finished batch's job outputs into the scan files fast mode
    reads; raise, naming them, while any job has no output."""
    bundle = Path(bundle).resolve()
    cfg, meta, masses = _bundle(bundle)
    rows, missing = [], []
    for i in range(masses.size):
        f = bundle / "dy_out" / f"dy_{i:04d}.txt"
        if f.exists():
            rows.append(np.loadtxt(f, ndmin=1))
        else:
            missing.append(i)
    if missing:
        shown = ", ".join(map(str, missing[:20])) + (", ..." if len(missing) > 20 else "")
        raise common.ToolchainError(
            f"{len(missing)} of {masses.size} Drell-Yan jobs have no output in "
            f"{bundle / 'dy_out'} yet (index {shown}).")
    table = np.array(rows)
    odd = [int(r[0]) for r in table if int(r[6]) != meta["nevents"]]
    if odd:
        print(f"[mcpsim] warning: job(s) {', '.join(map(str, odd))} ran with a different "
              f"number of events than the bundle's {meta['nevents']} (a smoke test left "
              f"in dy_out/?); rerun them for a uniform scan.")
    return _write_scan(cfg, bundle, table[:, 1], table[:, 2], table[:, 3])


_DY_JOB_LAUNCHER = '''\
"""One job of this mcpsim Drell-Yan bundle, or the assembly of its results:
    python3 dy_job.py job <bundle> <index> [--cores N] [--nevents N]
    python3 dy_job.py collect <bundle>
See README_condor.txt."""
import sys

# The mcpsim source checkout that wrote this bundle, when it was one, so the
# jobs need no install; an installed mcpsim is used if the path is absent.
{sys_path}
try:
    from mcpsim.generation.madgraph import main
except ImportError as exc:
    sys.exit(f"cannot import mcpsim ({{exc}}); see README_condor.txt, step 3.")
main(sys.argv[1:])
'''

# Per-job wrapper, modelled on the brem bundle's run_brem_job.sh: cd to the
# bundle on the shared filesystem by absolute path (a worker runs the
# executable from its own scratch copy), source a CERN LCG view (compilers
# and LHAPDF for MadGraph) with `set -u` off, then run dy_job.py.
_DY_CONDOR_WRAPPER = """\
#!/usr/bin/env bash
# One Drell-Yan job: MadGraph for the mass on line <index> of dy_masses.txt
# (README_condor.txt). Condor jobs do not see the submitting shell's
# environment: change the defaults here, or set them in dy.sub with
#   environment = "MG5_DIR=/path PYTHON_BIN=/path"
set -euo pipefail
JOB_INDEX="${{1:?job index required}}"
NCPUS="${{2:-{n_proc}}}"
MG5_DIR="${{MG5_DIR:-{mg5_dir}}}"

WORKDIR="${{DY_WORKDIR:-{workdir}}}"
[[ -d "$WORKDIR" ]] || WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p condor_logs dy_out

for setup in \\
  /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh \\
  /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
do
  if [[ -f "$setup" ]]; then
    set +u
    source "$setup"
    set -u
    break
  fi
done

if [[ -n "$MG5_DIR" && -x "$MG5_DIR/bin/mg5_aMC" ]]; then
  export PATH="$MG5_DIR/bin:$PATH"
fi
if ! command -v mg5_aMC >/dev/null 2>&1; then
  echo "mg5_aMC not found: set MG5_DIR (README_condor.txt, step 1)" >&2
  exit 2
fi
# A patched copy of the PDF set in lhapdf/ comes first (README_condor.txt, step 2).
if [[ -f "$WORKDIR/lhapdf/pdfsets.index" ]]; then
  export LHAPDF_DATA_PATH="$WORKDIR/lhapdf${{LHAPDF_DATA_PATH:+:$LHAPDF_DATA_PATH}}"
fi

# Python: PYTHON_BIN, the active venv, dy_venv/ in this bundle, else the
# view's python3. A venv runs with -E so that the view's PYTHONPATH cannot
# shadow its packages; MadGraph, started from it, keeps the view's environment.
PYFLAGS=-E
if [[ -z "${{PYTHON_BIN:-}}" ]]; then
  if [[ -n "${{VIRTUAL_ENV:-}}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$WORKDIR/dy_venv/bin/python" ]]; then
    PYTHON_BIN="$WORKDIR/dy_venv/bin/python"
  else
    PYTHON_BIN=python3
    PYFLAGS=
  fi
fi

echo "job ${{JOB_INDEX}} on $(hostname), ${{NCPUS}} CPU(s), python ${{PYTHON_BIN}}, $(command -v mg5_aMC)"
"$PYTHON_BIN" $PYFLAGS dy_job.py job "$WORKDIR" "$JOB_INDEX" --cores "$NCPUS" \\
  ${{DY_NEVENTS:+--nevents "$DY_NEVENTS"}}
"""

_DY_CONDOR_SUBMIT = """\
universe       = vanilla
executable     = run_dy_job.sh
arguments      = $(ProcId) {n_proc}

output         = condor_logs/dy.$(ClusterId).$(ProcId).out
error          = condor_logs/dy.$(ClusterId).$(ProcId).err
log            = condor_logs/dy.$(ClusterId).log

request_cpus   = {n_proc}
request_memory = 2GB
request_disk   = 2GB
+JobFlavour    = "workday"

# One job per mass in dy_masses.txt (index = ProcId).
queue {n_masses}
"""

_DY_CONDOR_README = """\
mcpsim Drell-Yan bundle: {n_masses} jobs, one per mass in dy_masses.txt, for
{name} ({energy:g} GeV beam): MadGraph LO q q~ > chi chi~ with {pdf_set}
for both beams, {nevents} events per mass. Run it from a shared filesystem
(AFS/EOS) on lxplus.cern.ch: the jobs cd to {workdir}
(override with DY_WORKDIR if you move the bundle), build their MadGraph
process in the worker's scratch space and write one line each to dy_out/.

Condor jobs do not see your shell's environment: the settings below go in
the defaults at the top of run_dy_job.sh, or in dy.sub as
    environment = "MG5_DIR=/path/to/MG5_aMC PYTHON_BIN=/path/to/python"

1. MadGraph5_aMC 3.x on AFS/EOS. {mg5_line}
   The jobs source an LCG view first; its lhapdf-config is what MadGraph's
   default setting (lhapdf_py3 = lhapdf-config) finds.

2. The PDF set {pdf_set} (LHAPDF id {lhaid}). LHAPDF >= 6.5 rejects its
   .info file, which predates two keys it now requires, so give the jobs a
   patched copy in lhapdf/ here (used first). Once:
     source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
     mkdir -p lhapdf && cd lhapdf
     cp "$(lhapdf-config --datadir)/pdfsets.index" .
     curl -L https://lhapdfsets.web.cern.ch/current/{pdf_set}.tar.gz | tar xz
     grep -q AlphaS_NumFlavors {pdf_set}/{pdf_set}.info ||
       printf '\\nAlphaS_FlavorScheme: variable\\nAlphaS_NumFlavors: 5\\n' >> {pdf_set}/{pdf_set}.info
     cd ..

3. Python with mcpsim. {python_step}

4. Smoke test one mass interactively (1000 events, 2 cores), then remove it:
     DY_NEVENTS=1000 ./run_dy_job.sh 0 2
     rm -rf work dy_out/dy_0000.txt dy_out/banner_0000.txt

5. Submit:  condor_submit dy.sub     Logs: condor_logs/

6. Outputs: dy_out/dy_<index>.txt (mass, sigma, acceptance, events, hits)
   and dy_out/banner_<index>.txt (MadGraph's run banner). The last job to
   finish assembles
     {dy_cross}
     {dy_ageo}
   ("python3 dy_job.py collect ." does it by hand and names any job without
   output). Use them in a run with this in your config:
     data:
       dy_cross: {dy_cross}
       dy_ageo: {dy_ageo}
       dy_material: iron
"""


def main(argv: Optional[List[str]] = None) -> None:
    """The batch entry points: `job` (one mass) and `collect` (the scan)."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m mcpsim.generation.madgraph",
        description="Drell-Yan HTCondor bundle: run one job, or assemble the results.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    job = sub.add_parser("job", help="run MadGraph for one mass of a bundle")
    job.add_argument("bundle", type=Path)
    job.add_argument("index", type=int)
    job.add_argument("--cores", type=int, default=None, help="MadGraph cores (default: the bundle's)")
    job.add_argument("--nevents", type=int, default=None, help="events (default: the bundle's)")
    col = sub.add_parser("collect", help="assemble a finished bundle's outputs")
    col.add_argument("bundle", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "job":
            run_condor_job(args.bundle, args.index, cores=args.cores, nevents=args.nevents)
        else:
            for f in collect_condor_dy(args.bundle):
                print(f"wrote {f}")
    except common.ToolchainError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
