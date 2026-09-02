"""PYTHIA meson parent samples for the Python decay engine.

The vectorized decay engine (generation/pydecay.py) re-decays archived PYTHIA
parent mesons instead of re-running the generator, which makes acceptance for
a *new* detector geometry a minutes-scale computation. The archived samples
live in the source repo (mesongen-backup/output-data/debug*.root, 120 GeV
fixed-target, HardQCD pTHatMin=2) unless engine.samples_dir points elsewhere.

Per-meson defaults balance MC precision against runtime: high-statistics
samples (pi0, eta, omega) are subsampled, low-statistics ones (rho, phi,
jpsi) are re-decayed `repeats` times with clustered error accounting downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import paths
from ..config import Config
from . import common

Parents = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # px, py, pz, E


@dataclass(frozen=True)
class SampleSpec:
    """One archived parent sample: file name, parents to use, decay repeats."""
    file: str
    n_use: Optional[int]     # None -> all entries
    repeats: int


# The archived samples were generated at this beam energy; using them for a
# different beam would silently produce wrong kinematics.
SAMPLE_BEAM_GEV = 120.0

# Tuned for ~2-3% (8% for jpsi, limited by 4132 parents) clustered MC error
# on a ~4e-8 sr face at 1 km; see the FLAME off-axis study. The tuning covers
# 4-14 mrad off-axis; if you push past that, check the reported worst relerr
# (and the JSON sidecars) rather than assuming these counts still suffice.
DEFAULT_SAMPLES: Dict[str, SampleSpec] = {
    "pi0":   SampleSpec("debugPi0.root",   6_000_000, 1),
    "eta":   SampleSpec("debugEta.root",   4_000_000, 1),
    "rho":   SampleSpec("debugRHO.root",   None,      7),
    "omega": SampleSpec("debugOmega.root", 4_000_000, 1),
    "phi":   SampleSpec("debugPhi.root",   None,      18),
    "jpsi":  SampleSpec("debugJsi.root",   None,      600),
}


def samples_dir(cfg: Config) -> Path:
    """Directory holding the parent ROOT samples."""
    if cfg.engine.samples_dir:
        return Path(cfg.engine.samples_dir).expanduser()
    return paths.mesongen_dir() / "output-data"


def have_samples(cfg: Config, meson: str) -> bool:
    """True when the archived sample for `meson` is available AND applicable
    (the archives are 120 GeV; a different beam energy must not use them)."""
    spec = DEFAULT_SAMPLES.get(meson)
    if spec is None or not (samples_dir(cfg) / spec.file).exists():
        return False
    if (not cfg.beam.is_collider and cfg.engine.samples_dir is None
            and abs(cfg.beam.energy_gev - SAMPLE_BEAM_GEV) > 0.02 * SAMPLE_BEAM_GEV):
        return False
    return True


def load_parents(cfg: Config, meson: str, *, quick: bool = False
                 ) -> Tuple[Parents, int]:
    """Load parent 4-momenta for `meson`; returns ((px,py,pz,E), repeats).

    Handles both tree layouts found in the archives: (px,py,pz,e) directly,
    or (magnitude,theta,phi) polar form (reconstructed with the PDG mother
    mass). `quick` caps the sample at 200k parents and a quarter of the
    repeats for smoke tests.
    """
    if (not cfg.beam.is_collider
            and abs(cfg.beam.energy_gev - SAMPLE_BEAM_GEV) > 0.02 * SAMPLE_BEAM_GEV
            and cfg.engine.samples_dir is None):
        raise common.ToolchainError(
            f"the archived PYTHIA parent samples were generated at "
            f"{SAMPLE_BEAM_GEV:g} GeV; this config's beam is "
            f"{cfg.beam.energy_gev:g} GeV. Generate samples for this energy "
            f"(`mcpsim generate mesons --engine cpp`, needs PYTHIA) and point "
            f"engine.samples_dir at them."
        )
    common.require_module(
        "uproot",
        "uproot is a core dependency as of mcpsim 0.2 — `pip install -e .`.",
    )
    import uproot

    from ..physics.constants import MESONS

    spec = DEFAULT_SAMPLES.get(meson)
    if spec is None:
        raise common.ToolchainError(
            f"no archived parent sample is defined for meson '{meson}' "
            f"(known: {', '.join(DEFAULT_SAMPLES)})."
        )
    path = samples_dir(cfg) / spec.file
    if not path.exists():
        raise common.ToolchainError(
            f"parent sample {path} not found. Install the PYTHIA sample bundle "
            f"or set engine.samples_dir, or run `mcpsim generate mesons` to "
            f"produce fresh samples."
        )

    tree = uproot.open(path)["mesons"]
    n_avail = tree.num_entries
    n = min(spec.n_use or n_avail, n_avail)
    repeats = spec.repeats
    if quick:
        n = min(n, 200_000)
        repeats = max(1, repeats // 4)

    branches = set(tree.keys())
    if {"px", "py", "pz", "e"} <= branches:
        arr = tree.arrays(["px", "py", "pz", "e"], entry_stop=n, library="np")
        parents = tuple(arr[b].astype(np.float64) for b in ("px", "py", "pz", "e"))
    elif {"magnitude", "theta", "phi"} <= branches:
        arr = tree.arrays(["magnitude", "theta", "phi"], entry_stop=n, library="np")
        p = arr["magnitude"].astype(np.float64)
        th = arr["theta"].astype(np.float64)
        ph = arr["phi"].astype(np.float64)
        mass = MESONS[meson].mass
        st = np.sin(th)
        parents = (p * st * np.cos(ph), p * st * np.sin(ph), p * np.cos(th),
                   np.hypot(p, mass))
    else:
        raise common.ToolchainError(
            f"unrecognised branch layout in {path}: {sorted(branches)} "
            f"(expected px/py/pz/e or magnitude/theta/phi)."
        )
    return parents, repeats
