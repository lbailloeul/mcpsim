"""Configuration model: beam, target, detector, sensitivity, and data references.

A config can be a built-in preset (darkquest / ship / lanl_12bar) or a user YAML
file. Minimal user configs (just beam/target/detector) inherit a sensible base
preset chosen from the beam energy, so fast mode works out of the box for a new
detector geometry at a known beam energy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import paths
from .physics import constants as C

DEFAULT_CHANNELS = ["meson_decay", "brem", "drell_yan"]


class _FloatLoader(yaml.SafeLoader):
    """SafeLoader that resolves scientific notation without a sign (e.g. 1.0e20).

    Stock PyYAML only matches floats like 1.0e+20; '1.0e20' would otherwise load
    as a string and silently break numeric config values (e.g. n_pot).
    """


_FloatLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
             [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


@dataclass
class BeamConfig:
    """Beam settings. energy_gev must sit within 2% of a tabulated meson-
    multiplicity beam (120/400/0.8 GeV) unless data.c_meson is supplied.
    In collider frame energy_gev is beam A (or the CM energy for PYTHIA
    regeneration) and brem/Drell-Yan are disabled."""
    energy_gev: float = 120.0
    frame: str = "fixed_target"      # 'fixed_target' | 'collider'
    n_pot: float = 1.0e20
    energy_b_gev: float = 0.0        # second beam energy (collider frame)

    @property
    def is_collider(self) -> bool:
        return self.frame == "collider"


@dataclass
class TargetConfig:
    """Target material. Enters the bremsstrahlung effective luminosity only;
    per-POT meson multiplicities are tabulated independent of material."""
    name: str = "iron"
    Z: int = 26
    A: float = 55.845
    density_g_cm3: float = 7.87
    interaction_length_cm: float = 16.8

    @classmethod
    def from_dict(cls, d: dict) -> "TargetConfig":
        """Resolve a target spec; a bare {name: ...} is filled from the table.

        An unrecognised material name must come with the full {Z, A,
        density_g_cm3, interaction_length_cm} set — previously it silently
        inherited iron's numbers, which is wrong physics with no warning.
        """
        allowed = {"name", "Z", "A", "density_g_cm3", "interaction_length_cm"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(
                f"unknown key(s) {sorted(unknown)} in the 'target' section; "
                f"valid keys: {sorted(allowed)}"
            )
        name = d.get("name", "iron")
        mat = C.TARGET_MATERIALS.get(name)
        overrides = {k: d[k] for k in ("Z", "A", "density_g_cm3", "interaction_length_cm")
                     if d.get(k) is not None}
        if mat is None and len(overrides) < 4:
            raise ValueError(
                f"unknown target material '{name}' (known: "
                f"{', '.join(sorted(C.TARGET_MATERIALS))}). Either use a known "
                f"name or supply all of Z, A, density_g_cm3, "
                f"interaction_length_cm for a custom material."
            )
        base = dict(
            name=name,
            Z=mat.Z if mat else 26,
            A=mat.A if mat else 55.845,
            density_g_cm3=mat.density_g_cm3 if mat else 7.87,
            interaction_length_cm=mat.interaction_length_cm if mat else 16.8,
        )
        base.update(overrides)
        return cls(**base)


@dataclass
class DetectorConfig:
    """Detector geometry.

    offaxis_mrad displaces the face centre transversely from the beam axis by
    distance_m * tan(offaxis_mrad/1000); acceptance is then computed with the
    azimuthal-average kernel (geometry.face_hit_fraction). Only the Python
    decay engine supports a nonzero offset.
    """
    type: str = "cylindrical"        # 'cylindrical' | 'bar_array'
    distance_m: float = 40.0
    radius_m: float = 0.5            # cylindrical
    bar_columns: int = 3            # bar_array: bars per row (face width)
    bar_rows: int = 2                #            bars per column (face height)
    bar_layers: int = 2              #            layers in depth (coincidence)
    bar_size_m: float = 0.05         # transverse bar cross-section
    bar_length_m: Optional[float] = None  # bar length along the line of sight;
                                     # enters n_gamma via sensitivity.scintillator
    face_mode: str = "rectangular"
    offaxis_mrad: float = 0.0        # transverse off-axis angle of the face centre


@dataclass
class EngineConfig:
    """How meson-decay acceptance is computed when it must be (re)derived.

    decay:    'python' — vectorized re-decay of archived PYTHIA parents
              (fast, any geometry incl. off-axis, no ROOT needed);
              'cpp'    — the legacy C++ binaries (lanl_decayPion_12bar.cc,
              decayVectorMeson.cc), kept for cross-checks.
    fidelity: 'corrected' — physically-correct samplers (default);
              'legacy'    — bit-for-bit reproduction of the published pipeline.
    """
    decay: str = "python"
    fidelity: str = "corrected"
    samples_dir: Optional[str] = None   # override the archived-sample location
    seed: int = 20260727
    quick: bool = False                 # cap sample sizes (smoke tests; not cached)

    def __post_init__(self) -> None:
        if self.decay not in ("python", "cpp"):
            raise ValueError(f"engine.decay must be 'python' or 'cpp', got {self.decay!r}")
        if self.fidelity not in ("corrected", "legacy"):
            raise ValueError(
                f"engine.fidelity must be 'corrected' or 'legacy', got {self.fidelity!r}")


@dataclass
class SensitivityConfig:
    """Detection model D(eps) = (1 - exp(-n_gamma eps^2))^a * eps^2 and the
    exclusion criterion. n_gamma = photoelectrons at eps = 1, a = layer
    coincidence, n_chi_threshold = signal events at the limit (3.09 = 95% CL
    for zero background). charge_* define the log-spaced eps scan grid.

    Instead of a raw n_gamma, set `scintillator` (plastic | cebr, see
    constants.SCINTILLATORS) and detector.bar_length_m: n_gamma is then
    derived by linear scaling from the material's GEANT4 anchor value —
    e.g. plastic anchored at 2.5e5 PE for a 1.5 m bar. Setting both
    scintillator and an explicit n_gamma in the same config is an error."""
    n_gamma: float = 2.5e5
    scintillator: Optional[str] = None
    a: float = 3.0
    n_chi_threshold: float = 3.09
    charge_min: float = 1e-5
    charge_max: float = 1.0
    n_charge: int = 500


@dataclass
class DataConfig:
    """References to precomputed inputs used in fast mode."""
    acceptance: Dict[str, str] = field(default_factory=dict)  # meson -> file
    brem_dir: Optional[str] = None
    brem_pattern: str = "Brem_*.txt"
    dy_cross: Optional[str] = None
    dy_ageo: Optional[str] = None
    mass_grid: str = "mship_values.txt"
    c_meson: Optional[Dict[str, float]] = None
    upsilon_ageo: Optional[float] = None   # no legacy scan file exists; flat value


@dataclass
class Config:
    """A complete experiment description (the top-level YAML schema).

    See docs/usage.md for the field-by-field reference. Built by
    load_preset()/load_config(); user configs overlay a base preset (explicit
    `preset:` key, else inferred from the beam energy)."""
    name: str = "experiment"
    family: str = "darkquest"        # used for upsilon-acceptance + brem defaults
    source_repo: Optional[str] = None  # override github-repo-scripts location
    beam: BeamConfig = field(default_factory=BeamConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    channels: List[str] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    engine: EngineConfig = field(default_factory=EngineConfig)
    notes: List[str] = field(default_factory=list)  # config-time warnings, shown at run time
    # True while the detector geometry is exactly the base preset's, i.e. the
    # preset's precomputed acceptance files are valid. A user config that
    # overrides `detector:` flips this and acceptance is recomputed.
    geometry_is_preset: bool = True

    # -- derived helpers ----------------------------------------------------
    @property
    def tag(self) -> str:
        """Filesystem-safe slug of the experiment name (output subdirectory)."""
        return re.sub(r"\W+", "_", self.name.strip().lower()) or "experiment"

    @property
    def needs_source_repo(self) -> bool:
        """True when any fast-mode data reference is repo-relative.

        A fully self-contained config (absolute mass_grid/acceptance/brem/DY
        paths, explicit c_meson) can run without the source repo; the CLI only
        enforces the repo's presence when something actually resolves there.
        Only the ACTIVE channels' references count — an inherited preset's
        brem/DY paths are irrelevant when those channels are disabled."""
        refs = [self.data.mass_grid]
        if "meson_decay" in self.channels:
            refs += list(self.data.acceptance.values())
        if "brem" in self.channels:
            refs.append(self.data.brem_dir)
        if "drell_yan" in self.channels:
            refs += [self.data.dy_cross, self.data.dy_ageo]
        return any(r and not Path(str(r)).expanduser().is_absolute() for r in refs)

    def c_meson_rates(self) -> Dict[str, float]:
        """Per-POT meson multiplicities for this beam.

        Explicit config values win; otherwise look up the tabulated PYTHIA /
        Burman-Smith set for this beam energy. Raises if the energy is unknown
        and no rates were supplied (the caller should suggest --regenerate).
        """
        if self.data.c_meson:
            return dict(self.data.c_meson)
        key = _beam_key(self.beam.energy_gev)
        if key is not None:
            return dict(C.C_MESON_BY_BEAM[key])
        tabulated = ", ".join(f"{k} GeV" for k in C.C_MESON_BY_BEAM)
        raise ValueError(
            f"No tabulated meson multiplicities for beam energy "
            f"{self.beam.energy_gev} GeV (tabulated: {tabulated}; matched "
            f"within 2%). Provide data.c_meson in the config — a mapping of "
            f"per-POT meson multiplicities from your generator, e.g. "
            f"data: {{c_meson: {{pi0: 4.7, eta: 0.53, ...}}}}."
        )

    def upsilon_ageo(self) -> Optional[float]:
        """Flat upsilon acceptance: explicit data.upsilon_ageo, else the
        per-family legacy default, else None (channel is skipped with a
        warning — previously an unknown family silently got DarkQuest's value)."""
        if self.data.upsilon_ageo is not None:
            return self.data.upsilon_ageo
        return C.UPSILON_AGEO_DEFAULT.get(self.family)


def _beam_key(energy_gev: float) -> Optional[str]:
    """Key into C_MESON_BY_BEAM if the energy matches a tabulated beam within
    2%, else None (previously any energy was rounded to the nearest GeV, so
    e.g. 120.5 GeV silently reused the 120 GeV multiplicities)."""
    for key in C.C_MESON_BY_BEAM:
        tab = float(key)
        if abs(energy_gev - tab) <= 0.02 * tab:
            return key
    return None


# -- loading ----------------------------------------------------------------
def list_presets() -> List[str]:
    if not paths.PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in paths.PRESETS_DIR.glob("*.yaml"))


def load_preset(name: str) -> Config:
    path = paths.PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown preset '{name}'. Available: {', '.join(list_presets()) or '(none)'}"
        )
    return _config_from_dict(_read_yaml(path))


def load_config(path: str | Path) -> Config:
    """Load a user YAML config, inheriting a base preset where appropriate."""
    raw = _read_yaml(Path(path))
    base_name = raw.get("preset")
    inferred = base_name is None
    if inferred:
        base_name = _infer_base_preset(raw)
    elif base_name not in list_presets():
        raise ValueError(
            f"unknown preset '{base_name}' referenced by the 'preset:' key; "
            f"available: {', '.join(list_presets()) or '(none)'}"
        )
    base = load_preset(base_name) if base_name and base_name in list_presets() else Config()
    cfg = _config_from_dict(raw, base=base)

    # A user config that overrides the detector invalidates the base preset's
    # precomputed acceptance files; model._meson_ageo recomputes in that case.
    if "detector" in raw:
        cfg = replace(cfg, geometry_is_preset=False)

    # An inferred base preset drags its precomputed brem/DY/acceptance inputs
    # along; if the beam energy differs from the preset's, those inputs were
    # computed at the WRONG energy. Record it so the run report says so.
    if inferred and base_name and base.beam.energy_gev > 0:
        de = abs(cfg.beam.energy_gev - base.beam.energy_gev) / base.beam.energy_gev
        if de > 0.01:
            cfg = replace(cfg, notes=cfg.notes + [
                f"config inherits the '{base_name}' preset "
                f"({base.beam.energy_gev:g} GeV) for a {cfg.beam.energy_gev:g} GeV "
                f"beam: its precomputed brem/DY/acceptance inputs were computed "
                f"at {base.beam.energy_gev:g} GeV. Use --regenerate (or set "
                f"'preset:' explicitly) for correct inputs at this energy."
            ])
    return cfg


def _infer_base_preset(raw: dict) -> Optional[str]:
    energy = (raw.get("beam") or {}).get("energy_gev")
    if energy is None:
        return None
    if energy <= 1.5:
        return "lanl_12bar"
    if energy >= 300:
        return "ship"
    return "darkquest"


def _read_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.load(fh, Loader=_FloatLoader) or {}


def _config_from_dict(raw: dict, base: Optional[Config] = None) -> Config:
    """Build a Config from a dict, overlaying on an optional base preset."""
    cfg = base or Config()

    if "name" in raw:
        cfg = replace(cfg, name=raw["name"])
    if "family" in raw:
        cfg = replace(cfg, family=raw["family"])
    if "source_repo" in raw:
        cfg = replace(cfg, source_repo=raw["source_repo"])
        paths.set_source_repo(raw["source_repo"])
    if "channels" in raw:
        cfg = replace(cfg, channels=list(raw["channels"]))

    if "beam" in raw:
        cfg = replace(cfg, beam=replace(cfg.beam, **_clean(raw["beam"], BeamConfig, "beam")))
    if "target" in raw:
        cfg = replace(cfg, target=TargetConfig.from_dict(raw["target"]))
    if "detector" in raw:
        cfg = replace(cfg, detector=replace(cfg.detector, **_clean(raw["detector"], DetectorConfig, "detector")))
    if "sensitivity" in raw:
        cfg = replace(cfg, sensitivity=replace(cfg.sensitivity, **_clean(raw["sensitivity"], SensitivityConfig, "sensitivity")))
        sens_raw = raw["sensitivity"] or {}
        if "scintillator" in sens_raw:
            if "n_gamma" in sens_raw:
                raise ValueError(
                    "set either sensitivity.scintillator (n_gamma derived from "
                    "the material anchor and detector.bar_length_m) or an "
                    "explicit sensitivity.n_gamma — not both."
                )
            cfg = replace(cfg, sensitivity=replace(
                cfg.sensitivity,
                n_gamma=_scintillator_n_gamma(cfg.sensitivity.scintillator,
                                              cfg.detector.bar_length_m)))
    if "data" in raw:
        cfg = replace(cfg, data=replace(cfg.data, **_clean(raw["data"], DataConfig, "data")))
    if "engine" in raw:
        cfg = replace(cfg, engine=replace(cfg.engine, **_clean(raw["engine"], EngineConfig, "engine")))

    return cfg


def _scintillator_n_gamma(name: Optional[str], bar_length_m: Optional[float]) -> float:
    """n_gamma from a material anchor, scaled linearly with bar length.

    n_gamma = n_gamma_ref * bar_length_m / length_ref_m, with the per-material
    (n_gamma_ref, length_ref_m) GEANT4 anchors in constants.SCINTILLATORS
    (plastic: 2.5e5 @ 1.5 m; cebr: 5.0e6 @ 1.5 m)."""
    spec = C.SCINTILLATORS.get(name or "")
    if spec is None:
        raise ValueError(
            f"unknown scintillator {name!r}; known: "
            f"{', '.join(sorted(C.SCINTILLATORS))} (anchors live in "
            f"physics/constants.SCINTILLATORS — add new materials there)."
        )
    if bar_length_m is None or bar_length_m <= 0:
        raise ValueError(
            f"sensitivity.scintillator = '{name}' needs detector.bar_length_m "
            f"(the bar length along the line of sight) to derive n_gamma; "
            f"set it, or give an explicit sensitivity.n_gamma instead."
        )
    return spec.n_gamma_ref * bar_length_m / spec.length_ref_m


def _clean(d: dict, dc, section: str) -> dict:
    """Validate a config section against the dataclass dc.

    Unknown keys raise instead of being dropped: a typo like 'radius:' for
    'radius_m:' previously vanished silently and the run used the inherited
    preset geometry."""
    allowed = {f.name for f in dc.__dataclass_fields__.values()}
    unknown = set(d or {}) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in the '{section}' config "
            f"section; valid keys: {sorted(allowed)}"
        )
    return dict(d or {})
