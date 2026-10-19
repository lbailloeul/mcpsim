"""Meson generation backends.

Selects the generator by beam energy / frame:
  * high-energy fixed target or collider -> PYTHIA 8 (mesongen-backup/mesonGen.cc)
  * low-energy fixed target               -> Burman-Smith parametrization
                                             (lanl_12bar_pipeline/generatePion_12bar.cc)

PYTHIA is invalid at the ~sub-GeV LANSCE beam, hence the Burman-Smith fallback,
which only produces pi0/eta. Both generators are compiled on demand against ROOT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

from .. import paths
from ..config import Config
from ..physics.constants import LOW_ENERGY_MESONS
from . import common

# Beam KE below this (GeV) -> Burman-Smith instead of PYTHIA.
LOW_ENERGY_THRESHOLD_GEV = 5.0

# PYTHIA particle ids for the mesons we extract.
PYTHIA_MESON_ID = {
    "pi0": 111, "eta": 221, "jpsi": 443, "upsilon": 553,
    "rho": 113, "omega": 223, "phi": 333,
}

# PYTHIA processes. The archived parent samples are hard QCD with
# pTHatMin = 2 (docs/physics.md). Hard QCD all but never makes an upsilon at
# these energies, so its sample comes from the bottomonium processes, which
# diverge as pT -> 0 and so need a pTHat cut of their own (PYTHIA manual,
# Onia processes).
HARD_QCD = ("HardQCD:all = on", "PhaseSpace:pTHatMin = 2.0")


def bottomonium(pthat_min: float) -> Tuple[str, ...]:
    return ("Bottomonium:all = on", f"PhaseSpace:pTHatMin = {pthat_min:g}")


# Mother masses in MeV for the Burman-Smith generator argument.
MOTHER_MASS_MEV = {"pi0": 134.9768, "eta": 547.862}


def choose_backend(cfg: Config) -> str:
    """Return 'pythia' or 'burman_smith' for this configuration."""
    if cfg.beam.is_collider:
        return "pythia"
    return "burman_smith" if cfg.beam.energy_gev < LOW_ENERGY_THRESHOLD_GEV else "pythia"


def generate(cfg: Config, meson: str, work_dir: Path, *, trials: int = 10_000_000,
             n_cores: int = 1) -> Path:
    """Generate a meson source sample, seeded with cfg.engine.seed; returns
    the produced file path."""
    backend = choose_backend(cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    if backend == "burman_smith":
        return _generate_burman_smith(meson, work_dir, trials, cfg.engine.seed)
    return _generate_pythia(cfg, meson, work_dir, trials, cfg.engine.seed, n_cores)


def _generate_burman_smith(meson: str, work_dir: Path, trials: int, seed: int) -> Path:
    if meson not in LOW_ENERGY_MESONS:
        raise common.ToolchainError(
            f"Burman-Smith generator only supports {LOW_ENERGY_MESONS}; got '{meson}'. "
            f"Heavier mesons need a beam energy above {LOW_ENERGY_THRESHOLD_GEV} GeV (PYTHIA)."
        )
    src = paths.legacy_dir() / "generatePion_12bar.cc"
    binary = common.compile_cpp(src, paths.CACHE_DIR / "bin" / "generatePion_12bar")
    out = work_dir / f"{meson}_source.txt"
    common.run([binary, str(out), str(trials), str(seed), str(MOTHER_MASS_MEV[meson])])
    return out


def _generate_pythia(cfg: Config, meson: str, work_dir: Path, trials: int,
                     seed: int, n_cores: int) -> Path:
    return _run_mesongen(cfg, meson, work_dir, trials, seed, n_cores, HARD_QCD)


def _run_mesongen(cfg: Config, meson: str, work_dir: Path, trials: int, seed: int,
                  n_cores: int, processes: Sequence[str]) -> Path:
    """Run mesonGen for `meson` with PYTHIA `processes`; returns the directory
    of its ROOT outputs, one per thread. mesonGen reads beam.config and
    momentum.config from its working directory and writes
    ../output-data/mesons_seed<seed>_t<thread>.root."""
    if meson not in PYTHIA_MESON_ID:
        raise common.ToolchainError(f"unknown meson '{meson}' for PYTHIA generation.")
    src = paths.legacy_dir() / "mesonGen.cc"
    binary = common.compile_cpp(src, paths.CACHE_DIR / "bin" / "mesonGen", with_pythia=True)
    run_dir, out_dir = work_dir / "run", work_dir / "output-data"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    _write_beam_config(cfg, run_dir / "beam.config", processes)
    _copy_momentum_config(run_dir / "momentum.config")
    common.run(
        [binary, str(seed), str(n_cores), str(trials), str(PYTHIA_MESON_ID[meson])],
        cwd=run_dir,
    )
    return out_dir


def generate_upsilon_sample(cfg: Config, work_dir: Path, *, trials: int = 20_000,
                            pthat_min: float = 2.0, n_cores: int = 1) -> Path:
    """A PYTHIA sample of upsilon(1S) parents from the bottomonium processes,
    merged into the one ROOT file data.upsilon_sample names."""
    import numpy as np
    import uproot

    if choose_backend(cfg) != "pythia":
        raise common.ToolchainError(
            f"an upsilon sample needs PYTHIA, i.e. a beam above {LOW_ENERGY_THRESHOLD_GEV:g} GeV.")
    out_dir = _run_mesongen(cfg, "upsilon", work_dir, trials, cfg.engine.seed, n_cores,
                            bottomonium(pthat_min))
    columns = ("px", "py", "pz", "e")
    parts = [uproot.open(f)["mesons"].arrays(columns, library="np")
             for f in sorted(out_dir.glob("*.root"))]
    merged = {c: np.concatenate([p[c] for p in parts]) for c in columns} if parts else {}
    if not merged or merged["px"].size == 0:
        raise common.ToolchainError(
            f"PYTHIA made no upsilon(1S) in {trials} events (outputs in {out_dir}).")
    sample = work_dir / f"upsilon_{cfg.beam.energy_gev:g}GeV.root"
    with uproot.recreate(sample) as f:
        f["mesons"] = merged
    return sample


def _write_beam_config(cfg: Config, dest: Path, processes: Sequence[str] = ()) -> None:
    """PYTHIA beam settings for the requested frame/energy, then `processes`."""
    lines = ["Beams:idA = 2212", "Beams:idB = 2212"]
    if cfg.beam.is_collider:
        # energy_gev is interpreted as the CM energy in collider mode.
        lines += ["Beams:frameType = 1", f"Beams:eCM = {cfg.beam.energy_gev}"]
    else:
        lines += [
            "Beams:frameType = 2",
            f"Beams:eA = {cfg.beam.energy_gev}",
            "Beams:eB = 0.0",
        ]
    dest.write_text("\n".join(lines + list(processes)) + "\n")


def _copy_momentum_config(dest: Path) -> None:
    src = paths.legacy_dir() / "momentum.config"
    dest.write_text(src.read_text())
