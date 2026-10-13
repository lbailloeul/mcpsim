"""Bremsstrahlung regeneration via the VEGAS integrator (mCP_brem.py).

mCP_brem.py hardcodes the beam/target as module globals (DarkQuest 120 GeV iron).
To regenerate for a different beam energy or target we copy the script into this
project's work dir and patch those globals on the *copy* (the source repo is
never modified), then run it.

Two execution modes:
  * local (default)  -- one `mpirun` over the whole mass list. Good for a few
                        masses / a new beam energy on a subset.
  * Condor (--condor) -- emit a ready-to-submit batch bundle (one job per mass,
                        MPI within each job), following the repo's proven
                        lxplus_brem.sub / run_lxplus_brem.sh. For full-grid
                        runs on LXPLUS/HTCondor; README_condor.txt in the
                        bundle says how.

Requires (local): mpirun on PATH, and the mpi4py + vegas Python modules.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .. import paths
from ..config import Config
from ..physics.constants import M_PROTON
from . import common


# The canonical grids (DarkQuest-/SHiP-brem-backup) are the eVMD "full" mode,
# with the rho/omega resonances: 335x the script's default "off_resonance" at
# m_chi = 0.30 GeV, 5400x at 0.52 GeV, identical at 1 MeV. Pass it explicitly.
FORM_FACTOR_MODE = "full"


def regenerate_brem(cfg: Config, masses: np.ndarray, work_dir: Path, *,
                    n_proc: int = 4, nitn: int = 10, neval: int = 4000,
                    grid_size: int = 100) -> Path:
    """Run the VEGAS brem integrator for `masses`; return the output directory."""
    common.require_command("mpirun", "Install an MPI runtime (e.g. openmpi).")
    common.require_module("mpi4py", "pip install mcpsim[gen]")
    common.require_module("vegas", "pip install mcpsim[gen]")

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "brem_out"
    out_dir.mkdir(exist_ok=True)

    script = _prepare_patched_script(cfg, work_dir)
    mass_file = work_dir / "brem_masses.txt"
    np.savetxt(mass_file, np.atleast_1d(masses), fmt="%.6g")

    common.run(
        ["mpirun", "-n", str(n_proc), "python", str(script),
         "--mass-file", str(mass_file), "--mass-cutoff", "1e9",
         "--form-factor-mode", FORM_FACTOR_MODE,
         "--lambdas", "1.0,1.5,2.0", "--nitn", str(nitn), "--neval", str(neval),
         "--grid-size", str(grid_size), "--output-dir", str(out_dir), "--force"],
        cwd=work_dir,
    )
    return out_dir


# Per-job wrapper run by each Condor process. Follows the proven
# archived/condor-brem-mcp-production/run_lxplus_brem.sh: cd to the bundle on
# the shared filesystem (AFS/EOS) by absolute path (a worker runs the
# executable from its own scratch copy), source a CERN LCG view for
# OpenMPI with `set -u` off (its setup scripts read unset variables), then
# run a venv Python with mpi4py/numpy/vegas, one mass index per job.
_CONDOR_WRAPPER = """\
#!/usr/bin/env bash
set -euo pipefail
MASS_INDEX="${{1:?mass index required}}"
NCPUS="${{2:-{n_proc}}}"

WORKDIR="${{BREM_WORKDIR:-{workdir}}}"
[[ -d "$WORKDIR" ]] || WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p condor_logs brem_out

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

# A venv (PYTHON_BIN, the active one, or brem_venv/ in this bundle) must not
# see the LCG view's Python path; the view's own python3 is the fallback.
if [[ -z "${{PYTHON_BIN:-}}" ]]; then
  if [[ -n "${{VIRTUAL_ENV:-}}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$WORKDIR/brem_venv/bin/python" ]]; then
    PYTHON_BIN="$WORKDIR/brem_venv/bin/python"
  fi
fi
if [[ -n "${{PYTHON_BIN:-}}" ]]; then
  unset PYTHONHOME PYTHONPATH
else
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [m for m in ("mpi4py", "numpy", "vegas") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing)
                     + " (see README_condor.txt: make brem_venv or set PYTHON_BIN)")
PY

echo "mass index ${{MASS_INDEX}} on $(hostname), ${{NCPUS}} CPU(s), python ${{PYTHON_BIN}}"
ARGS=(--mass-index "$MASS_INDEX" --mass-file brem_masses.txt --mass-cutoff 1e9
      --form-factor-mode {mode} --lambdas 1.0,1.5,2.0
      --nitn "${{BREM_NITN:-{nitn}}}" --neval "${{BREM_NEVAL:-{neval}}}"
      --grid-size "${{BREM_GRID_SIZE:-{grid_size}}}" --output-dir brem_out --force)
if command -v mpirun >/dev/null 2>&1 && [[ "$NCPUS" -gt 1 ]]; then
  mpirun -np "$NCPUS" "$PYTHON_BIN" {script} "${{ARGS[@]}}"
else
  "$PYTHON_BIN" {script} "${{ARGS[@]}}"
fi
"""

_CONDOR_README = """\
mcpsim brem bundle: {n_masses} jobs, one per mass in brem_masses.txt, for
{name} ({energy:g} GeV on {target}). Run it from a shared filesystem
(AFS/EOS) on lxplus.cern.ch; the jobs cd to {workdir}
(override with BREM_WORKDIR if you move the bundle).

1. Python with mpi4py, numpy and vegas (once):
     source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
     python3 -m venv brem_venv && brem_venv/bin/pip install mpi4py numpy vegas
   (or use another Python: environment = "PYTHON_BIN=/path/to/python" in
   brem.sub; Condor jobs do not see your shell's environment)

2. Smoke test one mass interactively (tiny VEGAS settings):
     BREM_NITN=1 BREM_NEVAL=50 BREM_GRID_SIZE=5 ./run_brem_job.sh 0 2

3. Submit:  condor_submit brem.sub     Logs: condor_logs/

4. Outputs: brem_out/Brem_{energy:g}GeV_<mass>.txt. Once all jobs finish, use
   them in a run with this in your config:
     data:
       brem_dir: {out_dir}
       brem_pattern: "Brem_*.txt"
"""

_CONDOR_SUBMIT = """\
universe       = vanilla
executable     = run_brem_job.sh
arguments      = $(ProcId) {n_proc}

output         = condor_logs/brem.$(ClusterId).$(ProcId).out
error          = condor_logs/brem.$(ClusterId).$(ProcId).err
log            = condor_logs/brem.$(ClusterId).log

request_cpus   = {n_proc}
request_memory = 6GB
request_disk   = 4GB
+JobFlavour    = "tomorrow"

# One job per mass in brem_masses.txt (index = ProcId).
queue {n_masses}
"""


def prepare_condor_brem(cfg: Config, masses: np.ndarray, work_dir: Path, *,
                        n_proc: int = 8, nitn: int = 10, neval: int = 4000,
                        grid_size: int = 100, submit: bool = True) -> Path:
    """Emit (and optionally submit) an HTCondor bundle for the brem grid.

    Writes the patched mCP_brem copy, the mass list, a per-job wrapper and a .sub
    that queues one job per mass. Submits via condor_submit when available;
    otherwise prints the manual command. Returns the (async) output directory.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "condor_logs").mkdir(exist_ok=True)
    out_dir = work_dir / "brem_out"
    out_dir.mkdir(exist_ok=True)

    script = _prepare_patched_script(cfg, work_dir)
    masses = np.atleast_1d(masses)
    np.savetxt(work_dir / "brem_masses.txt", masses, fmt="%.6g")

    wrapper = work_dir / "run_brem_job.sh"
    wrapper.write_text(_CONDOR_WRAPPER.format(
        n_proc=n_proc, nitn=nitn, neval=neval, grid_size=grid_size, script=script.name,
        workdir=work_dir.resolve(), mode=FORM_FACTOR_MODE))
    wrapper.chmod(0o755)
    (work_dir / "brem.sub").write_text(_CONDOR_SUBMIT.format(
        n_proc=n_proc, n_masses=masses.size))
    (work_dir / "README_condor.txt").write_text(_CONDOR_README.format(
        n_masses=masses.size, name=cfg.name, energy=cfg.beam.energy_gev,
        target=cfg.target.name, workdir=work_dir.resolve(), out_dir=out_dir.resolve()))

    if submit and shutil.which("condor_submit"):
        common.run(["condor_submit", "brem.sub"], cwd=work_dir)
        print(f"[mcpsim] submitted {masses.size} Condor jobs; outputs -> {out_dir} "
              f"(re-run `mcpsim run` once they finish).")
    elif submit:
        # Submission was requested but there is no scheduler here. The bundle
        # is written and usable — but exiting 0 with a scrolled-away print
        # previously made this look like a successful submission.
        raise common.ToolchainError(
            f"Condor bundle written to {work_dir} ({masses.size} jobs) but "
            f"'condor_submit' is not on PATH — nothing was submitted. Submit "
            f"from a Condor node with: cd {work_dir} && condor_submit brem.sub "
            f"(or use --no-submit to silence this)."
        )
    else:
        print(f"[mcpsim] Condor bundle (--no-submit) written to {work_dir} "
              f"({masses.size} jobs); see README_condor.txt there. Submit with:\n"
              f"  cd {work_dir} && condor_submit brem.sub")
    return out_dir


def _generator_face(det) -> Tuple[int, float]:
    """(n_bar, bar_size) of the generator's square, on-axis face for `det`.

    mCP_brem.py models an n x n array of bars centred on the beam. A bar array
    maps directly; a disc maps to the square of the same half-width (the
    paper's r = 0.5 m disc -> the 20 x 5 cm face the canonical grids used)."""
    if det.offaxis_mrad:
        raise common.ToolchainError(
            "the brem generator (mCP_brem.py) puts the detector face on the beam "
            "axis; it cannot make grids for an off-axis detector.")
    if det.type == "bar_array":
        if det.bar_columns != det.bar_rows:
            raise common.ToolchainError(
                f"the brem generator models a square face of n x n bars; this "
                f"detector is {det.bar_columns} x {det.bar_rows}.")
        return det.bar_columns, det.bar_size_m
    bar_size = 0.05
    return max(1, round(2 * det.radius_m / bar_size)), bar_size


def _prepare_patched_script(cfg: Config, work_dir: Path) -> Path:
    """Copy mCP_brem.py into work_dir and patch the beam/target globals."""
    src = paths.mcp_brem_script()
    if not src.exists():
        raise common.ToolchainError(f"mCP_brem.py not found at {src}.")
    text = src.read_text()

    t = cfg.target
    sqrt_s = math.sqrt(2.0 * M_PROTON * (cfg.beam.energy_gev + M_PROTON))
    # lambda_int_gcm2 / rho == interaction_length_cm  -> keep that identity.
    lambda_gcm2 = t.interaction_length_cm * t.density_g_cm3
    n_bar, bar_size = _generator_face(cfg.detector)

    # (pattern, replacement, required, what) — the script then derives
    # det_angle = (n_bar * bar_size / 2) / z_det itself, the half-width of its
    # square face over the distance: 12.5 mrad for the 1 x 1 m face at 40 m,
    # exactly as the canonical grids were made.
    patches: List[Tuple[str, str, bool, str]] = [
        (r"(?m)^\s*s\s*=\s*\(?15\.06\)?\s*\*\*\s*2.*$", f"s = ({sqrt_s:.6f})**2", True,
         f"sqrt(s) = {sqrt_s:.4f} GeV"),
        (r"(?m)^(\s*N_POT\s*=\s*)\S+", rf"\g<1>{cfg.beam.n_pot:.6e}", True, f"N_POT = {cfg.beam.n_pot:.3g}"),
        (r"(?m)^(\s*rho_fe\s*=\s*)\S+", rf"\g<1>{t.density_g_cm3}", True, f"rho = {t.density_g_cm3} g/cm3"),
        (r"(?m)^(\s*A_fe\s*=\s*)\S+", rf"\g<1>{t.A}", True, f"A = {t.A} g/mol"),
        (r"(?m)^(\s*lambda_int_gcm2\s*=\s*)\S+", rf"\g<1>{lambda_gcm2:.6f}", True,
         f"lambda_int = {lambda_gcm2:.3f} g/cm2"),
        (r"(?m)^(\s*z_det\s*=\s*)\S+", rf"\g<1>{cfg.detector.distance_m}", True,
         f"z_det = {cfg.detector.distance_m:g} m"),
        (r"(?m)^(\s*bar_size\s*=\s*)\S+", rf"\g<1>{bar_size}", True, f"bar_size = {bar_size:g} m"),
        (r"(?m)^(\s*n_bar\s*=\s*)\S+", rf"\g<1>{n_bar}", True, f"n_bar = {n_bar}"),
    ]
    for pattern, repl, required, what in patches:
        text, n = re.subn(pattern, repl, text)
        if n:
            print(f"[mcpsim] patched brem global: {what}")
        elif required:
            raise common.ToolchainError(
                f"could not patch expected global in mCP_brem.py (pattern {pattern!r}); "
                f"the script layout may have changed."
            )

    # The output header is hard-coded to the DarkQuest iron values. The readers
    # rescale grids by the header's A (physics.brem.target_rescale), so the
    # header must name the nucleus sigma_pA was actually computed for.
    header_patches: List[Tuple[str, str, bool]] = [
        ('"A_Fe = 55.845 g/mol   # atomic mass of iron\\n"',
         f'"A_target = {t.A} g/mol   # atomic mass ({t.name})\\n"', True),
        ('"ρ_Fe = 7.87 g/cm³  # density of iron\\n"',
         f'"rho_target = {t.density_g_cm3} g/cm³  # density ({t.name})\\n"', False),
        ('"L_target = 16.8 cm # one nuclear interaction length of protons in iron)\\n"',
         f'"L_target = {t.interaction_length_cm} cm # one nuclear interaction length ({t.name})\\n"', False),
        ('"E_beam = 120 GeV   # beam energy\\n"',
         f'"E_beam = {cfg.beam.energy_gev:g} GeV   # beam energy\\n"', False),
        ('"N_POT   = 1.0e20   # protons on target\\n"',
         f'"N_POT   = {cfg.beam.n_pot:.1e}   # protons on target\\n"', False),
    ]
    for old, new, required in header_patches:
        if old in text:
            text = text.replace(old, new)
        elif required:
            raise common.ToolchainError(
                f"could not patch the grid header in mCP_brem.py ({old!r}); the "
                f"header layout may have changed, and a stale 'A_' line would make "
                f"mcpsim rescale the regenerated grids wrongly."
            )

    # Output files are named for 120 GeV whatever the beam: name them for it.
    old_name = 'f"Brem_120GeV_{m_chi:g}.txt"'
    if old_name not in text:
        raise common.ToolchainError(
            "could not patch the output file name in mCP_brem.py; the script "
            "layout may have changed.")
    text = text.replace(old_name, f'f"Brem_{cfg.beam.energy_gev:g}GeV_{{m_chi:g}}.txt"')

    dest = work_dir / "mCP_brem_patched.py"
    dest.write_text(text)
    return dest
