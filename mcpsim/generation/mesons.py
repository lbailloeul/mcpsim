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

# Mother masses in MeV for the Burman-Smith generator argument.
MOTHER_MASS_MEV = {"pi0": 134.9768, "eta": 547.862}


def choose_backend(cfg: Config) -> str:
    """Return 'pythia' or 'burman_smith' for this configuration."""
    if cfg.beam.is_collider:
        return "pythia"
    return "burman_smith" if cfg.beam.energy_gev < LOW_ENERGY_THRESHOLD_GEV else "pythia"


def generate(cfg: Config, meson: str, work_dir: Path, *, trials: int = 10_000_000,
             seed: int = 12345, n_cores: int = 1) -> Path:
    """Generate a meson source sample; returns the produced file path."""
    backend = choose_backend(cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    if backend == "burman_smith":
        return _generate_burman_smith(meson, work_dir, trials, seed)
    return _generate_pythia(cfg, meson, work_dir, trials, seed, n_cores)


def _generate_burman_smith(meson: str, work_dir: Path, trials: int, seed: int) -> Path:
    if meson not in LOW_ENERGY_MESONS:
        raise common.ToolchainError(
            f"Burman-Smith generator only supports {LOW_ENERGY_MESONS}; got '{meson}'. "
            f"Heavier mesons need a beam energy above {LOW_ENERGY_THRESHOLD_GEV} GeV (PYTHIA)."
        )
    src = paths.lanl_dir() / "generatePion_12bar.cc"
    binary = common.compile_cpp(src, paths.CACHE_DIR / "bin" / "generatePion_12bar")
    out = work_dir / f"{meson}_source.txt"
    common.run([binary, str(out), str(trials), str(seed), str(MOTHER_MASS_MEV[meson])])
    return out


def _generate_pythia(cfg: Config, meson: str, work_dir: Path, trials: int,
                     seed: int, n_cores: int) -> Path:
    if meson not in PYTHIA_MESON_ID:
        raise common.ToolchainError(f"unknown meson '{meson}' for PYTHIA generation.")
    src = paths.mesongen_dir() / "mesonGen.cc"
    binary = common.compile_cpp(src, paths.CACHE_DIR / "bin" / "mesonGen", with_pythia=True)

    _write_beam_config(cfg, work_dir / "beam.config")
    _copy_momentum_config(work_dir / "momentum.config")

    # mesonGen reads beam.config/momentum.config from CWD and writes ROOT output.
    common.run(
        [binary, str(seed), str(n_cores), str(trials), str(PYTHIA_MESON_ID[meson])],
        cwd=work_dir,
    )
    return work_dir


def _write_beam_config(cfg: Config, dest: Path) -> None:
    """Template PYTHIA beam settings for the requested frame/energy."""
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
    dest.write_text("\n".join(lines) + "\n")


def _copy_momentum_config(dest: Path) -> None:
    src = paths.mesongen_dir() / "momentum.config"
    if src.exists():
        dest.write_text(src.read_text())
    else:  # minimal hard-QCD configuration if the repo file is absent
        dest.write_text(
            "HardQCD:all = on\nPhotonParton:all = on\nPhaseSpace:pTHatMin = 2\n"
            "ParticleDecays:allowPhotonRadiation = on\n"
        )
