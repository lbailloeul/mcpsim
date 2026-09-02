"""Bremsstrahlung regeneration via the VEGAS integrator (mCP_brem.py).

mCP_brem.py hardcodes the beam/target as module globals (DarkQuest 120 GeV iron).
To regenerate for a different beam energy or target we copy the script into this
project's work dir and patch those globals on the *copy* (the source repo is
never modified), then run it.

Two execution modes:
  * local (default)  -- one `mpirun` over the whole mass list. Good for a few
                        masses / a new beam energy on a subset.
  * Condor (--condor) -- emit a ready-to-submit batch bundle (one job per mass,
                        MPI within each job), modeled on the repo's existing
                        lxplus_brem.sub. For full-grid runs on LXPLUS/HTCondor.

Requires (local): mpirun on PATH, and the mpi4py + vegas Python modules.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .. import geometry, paths
from ..config import Config
from ..physics.constants import M_PROTON
from . import common


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
         "--lambdas", "1.0,1.5,2.0", "--nitn", str(nitn), "--neval", str(neval),
         "--grid-size", str(grid_size), "--output-dir", str(out_dir), "--force"],
        cwd=work_dir,
    )
    return out_dir


# Per-job wrapper run by each Condor process. Mirrors the repo's run_lxplus_brem.sh:
# source a CERN LCG view for mpi4py/numpy/vegas, then run one mass index under MPI.
_CONDOR_WRAPPER = """\
#!/usr/bin/env bash
set -euo pipefail
MASS_INDEX="${{1:?mass index required}}"
NCPUS="${{2:-{n_proc}}}"
cd "$(dirname "$0")"
mkdir -p condor_logs brem_out

# Provide mpi4py/numpy/vegas. On LXPLUS a CERN LCG view works out of the box;
# override by activating your own venv before condor_submit (set PYTHON_BIN).
for setup in \\
  /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh \\
  /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
do
  [[ -f "$setup" ]] && {{ source "$setup"; break; }}
done
PYTHON_BIN="${{PYTHON_BIN:-python3}}"

mpirun -np "$NCPUS" "$PYTHON_BIN" {script} \\
  --mass-index "$MASS_INDEX" \\
  --mass-file brem_masses.txt --mass-cutoff 1e9 \\
  --lambdas 1.0,1.5,2.0 --nitn {nitn} --neval {neval} --grid-size {grid_size} \\
  --output-dir brem_out --force
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
        n_proc=n_proc, nitn=nitn, neval=neval, grid_size=grid_size, script=script.name))
    wrapper.chmod(0o755)
    (work_dir / "brem.sub").write_text(_CONDOR_SUBMIT.format(
        n_proc=n_proc, n_masses=masses.size))

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
              f"({masses.size} jobs). Submit with:\n"
              f"  cd {work_dir} && condor_submit brem.sub")
    return out_dir


def _prepare_patched_script(cfg: Config, work_dir: Path) -> Path:
    """Copy mCP_brem.py into work_dir and patch the beam/target globals."""
    src = paths.mcp_brem_script()
    if not src.exists():
        raise common.ToolchainError(f"mCP_brem.py not found at {src}.")
    text = src.read_text()

    t = cfg.target
    sqrt_s = math.sqrt(2.0 * M_PROTON * (cfg.beam.energy_gev + M_PROTON))
    det_angle = geometry.acceptance_theta_cut(cfg.detector)
    # lambda_int_gcm2 / rho == interaction_length_cm  -> keep that identity.
    lambda_gcm2 = t.interaction_length_cm * t.density_g_cm3

    patches: List[Tuple[str, str, bool]] = [
        (r"(?m)^\s*s\s*=\s*\(?15\.06\)?\s*\*\*\s*2.*$", f"s = ({sqrt_s:.6f})**2", True),
        (r"(?m)^(\s*N_POT\s*=\s*)\S+", rf"\g<1>{cfg.beam.n_pot:.6e}", True),
        (r"(?m)^(\s*rho_fe\s*=\s*)\S+", rf"\g<1>{t.density_g_cm3}", True),
        (r"(?m)^(\s*A_fe\s*=\s*)\S+", rf"\g<1>{t.A}", True),
        (r"(?m)^(\s*lambda_int_gcm2\s*=\s*)\S+", rf"\g<1>{lambda_gcm2:.6f}", True),
        (r"(?m)^(\s*z_det\s*=\s*)\S+", rf"\g<1>{cfg.detector.distance_m}", False),
        (r"(?m)^(\s*det_angle\s*=\s*).*$", rf"\g<1>{det_angle:.8f}", True),
    ]
    for pattern, repl, required in patches:
        text, n = re.subn(pattern, repl, text)
        if n:
            print(f"[mcpsim] patched brem global: {repl.strip()}")
        elif required:
            raise common.ToolchainError(
                f"could not patch expected global in mCP_brem.py (pattern {pattern!r}); "
                f"the script layout may have changed."
            )

    dest = work_dir / "mCP_brem_patched.py"
    dest.write_text(text)
    return dest
