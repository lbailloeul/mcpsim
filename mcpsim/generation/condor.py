"""HTCondor batches: one job per mass, run from a bundle directory on a shared
filesystem (AFS/EOS) and submitted from an lxplus node.

The brem and Drell-Yan bundles share this frame: the file names, the start of
every job, the submit description and submission. Each writes its own job
payload and README.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from . import common

# CERN LCG views every job tries in order (compilers, OpenMPI, LHAPDF); the
# READMEs tell users to source the first.
LCG_VIEWS = ("/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh",
             "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh")

# A worker runs the executable from its own scratch copy, so a job cds to the
# bundle by absolute path. The LCG setup scripts read unset variables, so
# `set -u` is off while one is sourced.
_ENTER = """\
WORKDIR="${{{var}:-{workdir}}}"
[[ -d "$WORKDIR" ]] || WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
mkdir -p condor_logs {out}

for setup in \\
{views}
do
  if [[ -f "$setup" ]]; then
    set +u
    source "$setup"
    set -u
    break
  fi
done"""

_SUBMIT = """\
universe       = vanilla
executable     = run_{kind}_job.sh
arguments      = $(ProcId) {n_proc}

output         = condor_logs/{kind}.$(ClusterId).$(ProcId).out
error          = condor_logs/{kind}.$(ClusterId).$(ProcId).err
log            = condor_logs/{kind}.$(ClusterId).log

request_cpus   = {n_proc}
request_memory = {memory}
request_disk   = {disk}
+JobFlavour    = "{flavour}"

# One job per mass in {kind}_masses.txt (index = ProcId).
queue {n_jobs}
"""


@dataclass(frozen=True)
class Batch:
    """One batch of `n_jobs` jobs. `kind` names its files: the job wrapper
    run_<kind>_job.sh, <kind>.sub, <kind>_masses.txt, the outputs in
    <kind>_out/ and the <KIND>_WORKDIR override of the bundle's location."""
    kind: str
    n_jobs: int
    n_proc: int
    memory: str
    disk: str
    flavour: str

    def make_dirs(self, work_dir: Path) -> Path:
        """The bundle's directory, condor_logs/ and <kind>_out/ (returned)."""
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "condor_logs").mkdir(exist_ok=True)
        out = work_dir / f"{self.kind}_out"
        out.mkdir(exist_ok=True)
        return out

    def enter(self, workdir: Path) -> str:
        """The shell lines every job starts with: cd to the bundle at
        `workdir`, then source an LCG view."""
        return _ENTER.format(var=f"{self.kind.upper()}_WORKDIR", workdir=workdir,
                             out=f"{self.kind}_out",
                             views=" \\\n".join(f"  {view}" for view in LCG_VIEWS))

    def write_job(self, work_dir: Path, script: str) -> None:
        job = work_dir / f"run_{self.kind}_job.sh"
        job.write_text(script)
        job.chmod(0o755)

    def write_submit(self, work_dir: Path) -> None:
        (work_dir / f"{self.kind}.sub").write_text(_SUBMIT.format(
            kind=self.kind, n_proc=self.n_proc, memory=self.memory, disk=self.disk,
            flavour=self.flavour, n_jobs=self.n_jobs))

    def submit(self, work_dir: Path, submit: bool, submitted: str) -> None:
        """condor_submit the bundle, printing `submitted` after it. Asked to
        submit with no condor_submit on PATH, raise: the bundle is usable, but
        exiting 0 with a scrolled-away print looked like a submission. With
        submit=False, say how to submit it."""
        sub = f"{self.kind}.sub"
        if submit and shutil.which("condor_submit"):
            common.run(["condor_submit", sub], cwd=work_dir)
            print(f"[mcpsim] submitted {self.n_jobs} Condor jobs; {submitted}")
        elif submit:
            raise common.ToolchainError(
                f"Condor bundle written to {work_dir} ({self.n_jobs} jobs) but "
                f"'condor_submit' is not on PATH — nothing was submitted. Submit "
                f"from a Condor node with: cd {work_dir} && condor_submit {sub} "
                f"(or use --no-submit to silence this)."
            )
        else:
            print(f"[mcpsim] Condor bundle (--no-submit) written to {work_dir} "
                  f"({self.n_jobs} jobs); see README_condor.txt there. Submit with:\n"
                  f"  cd {work_dir} && condor_submit {sub}")
