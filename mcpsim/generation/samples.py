"""PYTHIA meson parent samples for the Python decay engine.

The vectorized decay engine (generation/pydecay.py) re-decays archived PYTHIA
parent mesons instead of re-running the generator, which makes acceptance for
a *new* detector geometry a minutes-scale computation. The archived samples
live in the source repo (mesongen-backup/output-data/, fixed-target, HardQCD
pTHatMin=2) unless engine.samples_dir points elsewhere. Sets exist for the
120 and 400 GeV beams; see DEFAULT_SAMPLES.

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


# Archived parent samples, by beam energy. Using a sample from the wrong beam
# would silently produce wrong kinematics, so each set is keyed by the energy
# it was generated at and matched within 2% (the same rule as
# constants.C_MESON_BY_BEAM).
#
# n_use x repeats is tuned for ~2-3% clustered MC error (8% for jpsi, limited by
# its few thousand parents) on a ~4e-8 sr face at 1 km; see the FLAME off-axis
# study. The tuning covers 4-14 mrad off-axis; if you push past that, check the
# reported worst relerr (and the JSON sidecars) rather than assuming these
# counts still suffice. Both sets target the same effective decay count, so the
# 400 GeV entries -- a smaller archive -- carry proportionally more repeats.
#
# The 120 GeV set uses the higher-statistics samples: acceptance is a per-species
# ratio, so the trial counts behind them do not matter here. Per-POT
# multiplicities must NOT be derived from these files (see C_MESON_BY_BEAM).
DEFAULT_SAMPLES: Dict[str, Dict[str, SampleSpec]] = {
    "120": {
        "pi0":   SampleSpec("mesons_120GeV_pi0_highstat.root",   6_000_000, 1),
        "eta":   SampleSpec("mesons_120GeV_eta_highstat.root",   4_000_000, 1),
        "rho":   SampleSpec("mesons_120GeV_rho_highstat.root",   None,      7),
        "omega": SampleSpec("mesons_120GeV_omega_highstat.root", 4_000_000, 1),
        "phi":   SampleSpec("mesons_120GeV_phi_highstat.root",   None,      18),
        "jpsi":  SampleSpec("mesons_120GeV_jpsi_highstat.root",  None,      600),
    },
    "400": {
        "pi0":   SampleSpec("mesons_400GeV_pi0.root",   None, 16),
        "eta":   SampleSpec("mesons_400GeV_eta.root",   None, 24),
        "rho":   SampleSpec("mesons_400GeV_rho.root",   None, 4),
        "omega": SampleSpec("mesons_400GeV_omega.root", None, 4),
        "phi":   SampleSpec("mesons_400GeV_phi.root",   None, 9),
        "jpsi":  SampleSpec("mesons_400GeV_jpsi.root",  None, 297),
    },
}


def _sample_set(cfg: Config) -> Optional[Dict[str, SampleSpec]]:
    """The sample set matching this config's beam, or None if there is none.

    With engine.samples_dir set the user supplies their own files, so the beam
    guard is waived and the 120 GeV naming is used as the filename convention.
    """
    from ..config import _beam_key
    key = _beam_key(cfg.beam.energy_gev)
    if key in DEFAULT_SAMPLES:
        return DEFAULT_SAMPLES[key]
    return DEFAULT_SAMPLES["120"] if cfg.engine.samples_dir else None


def sample_spec(cfg: Config, meson: str) -> Optional[SampleSpec]:
    """The SampleSpec for `meson` at this config's beam, or None."""
    specs = _sample_set(cfg)
    return specs.get(meson) if specs else None


def samples_dir(cfg: Config) -> Path:
    """Directory holding the parent ROOT samples."""
    if cfg.engine.samples_dir:
        return Path(cfg.engine.samples_dir).expanduser()
    return paths.mesongen_dir() / "output-data"


def have_samples(cfg: Config, meson: str) -> bool:
    """True when a parent sample for `meson` exists AND matches this beam."""
    specs = _sample_set(cfg)
    if specs is None:
        return False
    spec = specs.get(meson)
    return spec is not None and (samples_dir(cfg) / spec.file).exists()


def load_parents(cfg: Config, meson: str, *, quick: bool = False
                 ) -> Tuple[Parents, int]:
    """Load parent 4-momenta for `meson`; returns ((px,py,pz,E), repeats).

    Handles both tree layouts found in the archives: (px,py,pz,e) directly,
    or (magnitude,theta,phi) polar form (reconstructed with the PDG mother
    mass). `quick` caps the sample at 200k parents and a quarter of the
    repeats for smoke tests.
    """
    specs = _sample_set(cfg)
    if specs is None:
        raise common.ToolchainError(
            f"no archived parent samples for a {cfg.beam.energy_gev:g} GeV beam "
            f"(available: {', '.join(DEFAULT_SAMPLES)} GeV, matched within 2%). "
            f"Generate samples for this energy (`mcpsim generate mesons`, needs "
            f"PYTHIA) and point engine.samples_dir at them."
        )
    common.require_module(
        "uproot",
        "uproot is a core dependency as of mcpsim 0.2 — `pip install -e .`.",
    )
    import uproot

    from ..physics.constants import MESONS

    spec = specs.get(meson)
    if spec is None:
        raise common.ToolchainError(
            f"no archived parent sample is defined for meson '{meson}' "
            f"(known: {', '.join(specs)})."
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
