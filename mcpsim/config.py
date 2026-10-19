"""Configuration model: beam, target, detector, sensitivity, and data references.

A config can be a built-in preset (darkquest / ship / lanl_12bar) or a user YAML
file. Minimal user configs (just beam/target/detector) inherit a sensible base
preset chosen from the beam energy, so fast mode works out of the box for a new
detector geometry at a known beam energy.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from . import paths
from .physics import constants as C

DEFAULT_CHANNELS = ["meson_decay", "brem", "drell_yan"]
# Short names accepted wherever channels are given (they match `mcpsim generate`).
CHANNEL_ALIASES = {"mesons": "meson_decay", "meson": "meson_decay",
                   "dy": "drell_yan", "drellyan": "drell_yan"}
TOP_LEVEL_KEYS = ("preset", "name", "family", "source_repo", "channels", "beam",
                  "target", "detector", "legacy_detector", "sensitivity", "data",
                  "engine")
# A run record's keys, in write order; "config" holds asdict(cfg) (run_record).
RUN_RECORD_KEYS = ("mcpsim_version", "data_source", "config", "drell_yan_cross_sections")
# Short names for the tabulated target materials (iron = the DarkQuest dump,
# molybdenum = the SHiP TZM target, carbon = NuMI/LBNF graphite).
TARGET_ALIASES = {"fe": "iron", "mo": "molybdenum", "moly": "molybdenum",
                  "tzm": "molybdenum", "w": "tungsten", "c": "carbon",
                  "graphite": "carbon"}
# Config.acceptance_made_for value of a file the user config names: it is used
# as it is, whatever detector the run models.
SUPPLIED_BY_CONFIG = "named by the config"
# A DetectorConfig, BeamConfig or EngineConfig holding any other value cannot
# be built, so code that branches on these fields needs no fallback.
DETECTOR_TYPES = ("cylindrical", "bar_array")
BEAM_FRAMES = ("fixed_target", "collider")
DECAY_ENGINES = ("python", "cpp")


@dataclass(frozen=True)
class FidelityRules:
    """What a fidelity mode decides, one rule per field. FIDELITY_RULES is the
    one place the modes are told apart; code reads cfg.engine.rules, and
    docs/physics.md sizes each difference."""
    published_detector: bool    # a preset models its legacy_detector (run_config)
    dalitz_pi: float            # pi in the Dalitz I3 integral (the scripts used 3.14)
    cpp_dalitz_sampler: bool    # pydecay replays the C++ Dalitz kinematics
    exact_bar_array_brem: bool  # on-axis bar arrays get the face kernel, not the cone cut
    dy_as_published: bool       # the paper's Drell-Yan scan, with 13 mb for every target
    marked: bool                # runs carry the mode's name (the CLI's "<name> legacy")


FIDELITY_RULES: Dict[str, FidelityRules] = {
    "corrected": FidelityRules(published_detector=False, dalitz_pi=math.pi,
                               cpp_dalitz_sampler=False, exact_bar_array_brem=True,
                               dy_as_published=False, marked=False),
    "legacy": FidelityRules(published_detector=True, dalitz_pi=3.14,
                            cpp_dalitz_sampler=True, exact_bar_array_brem=False,
                            dy_as_published=True, marked=True),
}
FIDELITIES = tuple(FIDELITY_RULES)


_TARGET_NUMBERS = ("Z", "A", "density_g_cm3", "interaction_length_cm")


def _tabulated_name(name: object) -> Optional[str]:
    """The TARGET_MATERIALS key a material name or alias means, in any case
    and with surrounding space, or None. The one name rule behind a YAML
    `target.name` and the --target flag."""
    key = str(name).strip().lower()
    canon = TARGET_ALIASES.get(key, key)
    return canon if canon in C.TARGET_MATERIALS else None


def _unknown_target_message(name: object) -> str:
    key = str(name).strip().lower()
    return (f"unknown target material '{name}'"
            f"{_did_you_mean(key, list(C.TARGET_MATERIALS) + list(TARGET_ALIASES))} "
            f"(known: {', '.join(sorted(C.TARGET_MATERIALS))}; aliases: "
            f"{', '.join(TARGET_ALIASES)}). Either use a known name or give a "
            f"config's target section all of {', '.join(_TARGET_NUMBERS)} for a "
            f"custom material.")


def normalize_target(name: str) -> str:
    """Canonical tabulated material for a name or alias; unknown names raise."""
    canon = _tabulated_name(name)
    if canon is None:
        raise ValueError(_unknown_target_message(name))
    return canon


def _did_you_mean(word: str, choices: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.5)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _check_choice(field_name: str, value: object, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{field_name} must be {' or '.join(map(repr, choices))}, got "
                         f"{value!r}{_did_you_mean(str(value), choices)}")


def normalize_channels(channels: Iterable[str]) -> List[str]:
    """Map aliases to canonical channel names; an unknown name raises (it used
    to run with no yield at all and write empty plots)."""
    out = []
    for ch in channels:
        name = CHANNEL_ALIASES.get(str(ch).strip().lower(), str(ch).strip())
        if name not in DEFAULT_CHANNELS:
            raise ValueError(
                f"unknown channel '{ch}'{_did_you_mean(name, DEFAULT_CHANNELS)}; "
                f"valid: {', '.join(DEFAULT_CHANNELS)} (aliases: mesons, dy)")
        if name not in out:
            out.append(name)
    return out


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

    def __post_init__(self) -> None:
        _check_choice("beam.frame", self.frame, BEAM_FRAMES)

    @property
    def is_collider(self) -> bool:
        return self.frame == "collider"


@dataclass
class TargetConfig:
    """Target material. Enters the brem channel through the effective
    luminosity and the sigma_pA rescale of the grids (physics.brem); per-POT
    meson multiplicities are tabulated independent of material."""
    name: str = "iron"
    Z: int = 26
    A: float = 55.845
    density_g_cm3: float = 7.87
    interaction_length_cm: float = 16.8

    def overlay(self, spec: dict) -> "TargetConfig":
        """This target with a `target:` spec applied; a YAML section and the
        --target flag both come through here.

        A spec without a name keeps this material, so `preset: ship` with
        `target: {density_g_cm3: 8.0}` stays molybdenum. A tabulated name or
        alias starts from its table entry, and any other name is a custom
        material that must give all four numbers. The given numbers apply last.
        """
        allowed = {"name", *_TARGET_NUMBERS}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(
                f"unknown key(s) {sorted(unknown)} in the 'target' section; "
                f"valid keys: {sorted(allowed)}"
            )
        given = {k: spec[k] for k in _TARGET_NUMBERS if spec.get(k) is not None}
        name = spec.get("name")
        if name is None:
            return replace(self, **given)
        canon = _tabulated_name(name)
        if canon is not None:
            m = C.TARGET_MATERIALS[canon]
            return replace(TargetConfig(canon, m.Z, m.A, m.density_g_cm3,
                                        m.interaction_length_cm), **given)
        if len(given) < len(_TARGET_NUMBERS):
            raise ValueError(_unknown_target_message(name))
        return TargetConfig(name=name, **given)


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

    def __post_init__(self) -> None:
        _check_choice("detector.type", self.type, DETECTOR_TYPES)


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
        _check_choice("engine.decay", self.decay, DECAY_ENGINES)
        _check_choice("engine.fidelity", self.fidelity, FIDELITIES)

    @property
    def rules(self) -> FidelityRules:
        return FIDELITY_RULES[self.fidelity]


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
    # Polar angle up to which the brem grids are valid: they carry the
    # acceptance of the detector they were generated for, so brem through any
    # part of the face beyond this angle is underestimated (warned at run time).
    brem_max_theta_mrad: Optional[float] = None
    # The paper's MadGraph Drell-Yan scan (1-column pb, row-aligned with
    # dy_ageo). By default (fidelity corrected) the cross sections come from
    # the built-in per-material table for the target instead; this scan is
    # used by fidelity legacy, when a user config sets it, or when no table
    # exists for the target/beam energy (model.drell_yan_table).
    dy_cross: Optional[str] = None
    # Target material whose nuclear PDFs the dy_cross scan was simulated with.
    # The per-POT normalisation always follows the configured target; a
    # different material here is reported at run time.
    dy_material: Optional[str] = None
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
    # The detector of the published analysis, when it differs from `detector`
    # (DarkQuest/SHiP: the paper's r = 0.5 m disc vs the 1 x 1 m face mcpsim
    # models). The archived acceptance files were made for it, and fidelity
    # legacy models it instead of `detector` (see run_config).
    legacy_detector: Optional[DetectorConfig] = None
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    channels: List[str] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    engine: EngineConfig = field(default_factory=EngineConfig)
    notes: List[str] = field(default_factory=list)  # config-time warnings, shown at run time
    # True while the detector geometry is exactly the base preset's, i.e. the
    # preset's precomputed acceptance files are valid. A user config that
    # overrides `detector:` flips this and acceptance is recomputed.
    geometry_is_preset: bool = True
    # True while data.dy_cross is the base preset's scan, which the built-in
    # per-material Drell-Yan table replaces by default. A user config that
    # sets data.dy_cross flips this and its own scan is used.
    dy_cross_is_preset: bool = True
    # What each acceptance file the config supplied (a data.acceptance value
    # or data.dy_ageo, keyed by the path as data holds it) was made for: the
    # geometry.face_signature of the detector regeneration wrote it for, or
    # SUPPLIED_BY_CONFIG. A file not listed is the base preset's
    # (model.acceptance_file_valid). Written by mark_acceptance_supplied.
    acceptance_made_for: Dict[str, str] = field(default_factory=dict)

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
        key = next((k for k in C.C_MESON_BY_BEAM
                    if C.same_beam(self.beam.energy_gev, float(k))), None)
        if key is not None:
            return dict(C.C_MESON_BY_BEAM[key])
        tabulated = ", ".join(f"{k} GeV" for k in C.C_MESON_BY_BEAM)
        raise ValueError(
            f"No tabulated meson multiplicities for beam energy "
            f"{self.beam.energy_gev} GeV (tabulated: {tabulated}; matched "
            f"within {C.BEAM_MATCH:.0%}). Provide data.c_meson in the config — a "
            f"mapping of per-POT meson multiplicities from your generator, e.g. "
            f"data: {{c_meson: {{pi0: 4.7, eta: 0.53, ...}}}}."
        )

def run_record(cfg: Config, *, data_source: str,
               drell_yan_cross_sections: Optional[str] = None) -> dict:
    """The run_<tag>.yaml payload, keys in RUN_RECORD_KEYS order: provenance
    plus the resolved configuration as asdict(cfg). load_config reads it back
    (_config_from_run_record)."""
    from . import __version__
    record = {"mcpsim_version": __version__, "data_source": data_source,
              "config": asdict(cfg)}
    if "drell_yan" in cfg.channels:
        record["drell_yan_cross_sections"] = drell_yan_cross_sections
    return record


def _config_from_run_record(raw: dict) -> Config:
    """The Config a run record holds, exactly: no base preset, no overlay and
    no provenance re-derived (a record's detector: and data: would otherwise
    clear geometry_is_preset and mark every file as the config's own)."""
    unknown = set(raw) - set(RUN_RECORD_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} beside the 'config:' block of a run "
            f"record; valid keys: {', '.join(RUN_RECORD_KEYS)}. Change the run "
            f"under 'config:' instead.")
    try:
        cfg = config_from_record(raw["config"])
    except TypeError as exc:
        raise ValueError(
            f"this run record, written by mcpsim {raw.get('mcpsim_version', '?')}, "
            f"does not match this version's configuration: {exc}") from exc
    if cfg.source_repo:
        paths.set_source_repo(cfg.source_repo)
    return cfg


def config_from_record(record: dict) -> Config:
    """Rebuild a Config from dataclasses.asdict(cfg), as stored in run records
    and batch bundles (config.json): the inverse of asdict, with no preset
    overlay or re-validation."""
    sections = {"beam": BeamConfig, "target": TargetConfig, "detector": DetectorConfig,
                "sensitivity": SensitivityConfig, "data": DataConfig, "engine": EngineConfig}
    kw = dict(record)
    for key, dc in sections.items():
        if key in kw:
            kw[key] = dc(**kw[key])
    if kw.get("legacy_detector") is not None:
        kw["legacy_detector"] = DetectorConfig(**kw["legacy_detector"])
    return Config(**kw)


def mark_acceptance_supplied(cfg: Config, files: Iterable[str], made_for: str) -> Config:
    """cfg with `files` recorded as made for `made_for`: a detector's
    geometry.face_signature, or SUPPLIED_BY_CONFIG. Only files cfg.data uses
    are recorded, and records of files it no longer uses are dropped."""
    used = {*cfg.data.acceptance.values(), cfg.data.dy_ageo} - {None}
    stamps = {p: s for p, s in cfg.acceptance_made_for.items() if p in used}
    stamps.update({p: made_for for p in files if p in used})
    return replace(cfg, acceptance_made_for=stamps)


def pin_supplied_acceptance(cfg: Config) -> Config:
    """cfg with the files the user config names recorded as made for the
    detector cfg models now, so moving that detector (--offaxis) recomputes
    their acceptance instead of reusing them."""
    from . import geometry    # geometry imports this module
    named = [p for p, s in cfg.acceptance_made_for.items() if s == SUPPLIED_BY_CONFIG]
    return mark_acceptance_supplied(cfg, named, geometry.face_signature(cfg.detector))


def run_config(cfg: Config) -> Config:
    """The configuration a run models. fidelity = legacy on a preset that
    records the published detector (legacy_detector) models that detector, so
    the archived acceptance files and the brem cut reproduce the paper; every
    other run models `detector`. Idempotent."""
    if (cfg.engine.rules.published_detector and cfg.geometry_is_preset
            and cfg.legacy_detector is not None):
        return replace(cfg, detector=cfg.legacy_detector)
    return cfg


# -- loading ----------------------------------------------------------------
def list_presets() -> List[str]:
    if not paths.PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in paths.PRESETS_DIR.glob("*.yaml"))


def load_preset(name: str) -> Config:
    path = paths.PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown preset '{name}'{_did_you_mean(name, list_presets())}. "
            f"Available: {', '.join(list_presets()) or '(none)'}"
        )
    return _config_from_dict(_read_yaml(path))


def load_config(path: str | Path) -> Config:
    """Load a user YAML config, inheriting a base preset where appropriate, or
    a run record (run_<tag>.yaml, a top-level `config:` mapping), which reruns
    the recorded configuration exactly."""
    raw = _read_yaml(Path(path))
    if isinstance(raw.get("config"), dict):
        return _config_from_run_record(raw)
    base_name = raw.get("preset")
    inferred = base_name is None
    if inferred:
        base_name = _infer_base_preset(raw)
    elif base_name not in list_presets():
        raise ValueError(
            f"unknown preset '{base_name}' referenced by the 'preset:' key"
            f"{_did_you_mean(str(base_name), list_presets())}; "
            f"available: {', '.join(list_presets()) or '(none)'}"
        )
    base = load_preset(base_name) if base_name and base_name in list_presets() else Config()
    cfg = _provenance(_config_from_dict(raw, base=base), raw)

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


def _provenance(cfg: Config, raw: dict) -> Config:
    """cfg with the provenance a user config earns by what it sets (a preset
    earns none: its inputs are its own).

        detector:                    geometry_is_preset = False, so the base
                                     preset's acceptance files are recomputed
        data.dy_cross:               dy_cross_is_preset = False
        data.acceptance, dy_ageo:    the files named are SUPPLIED_BY_CONFIG
    """
    data = raw.get("data") or {}
    if "detector" in raw:
        cfg = replace(cfg, geometry_is_preset=False)
    if "dy_cross" in data:
        cfg = replace(cfg, dy_cross_is_preset=False)
    named = [*(data.get("acceptance") or {}).values(), data.get("dy_ageo")]
    return mark_acceptance_supplied(cfg, named, SUPPLIED_BY_CONFIG)


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

    # A misspelled section ('detecor:') used to be ignored silently, leaving
    # the inherited preset values in force.
    unknown = set(raw) - set(TOP_LEVEL_KEYS)
    if unknown:
        hint = "".join(_did_you_mean(k, TOP_LEVEL_KEYS) for k in sorted(unknown))
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)}{hint}; "
                         f"valid keys: {', '.join(TOP_LEVEL_KEYS)}")

    if "name" in raw:
        cfg = replace(cfg, name=raw["name"])
    if "family" in raw:
        cfg = replace(cfg, family=raw["family"])
    if "source_repo" in raw:
        cfg = replace(cfg, source_repo=raw["source_repo"])
        paths.set_source_repo(raw["source_repo"])
    if "channels" in raw:
        cfg = replace(cfg, channels=normalize_channels(raw["channels"]))

    if "beam" in raw:
        cfg = replace(cfg, beam=replace(cfg.beam, **_clean(raw["beam"], BeamConfig, "beam")))
    if "target" in raw:
        cfg = replace(cfg, target=cfg.target.overlay(raw["target"] or {}))
    if "detector" in raw:
        cfg = replace(cfg, detector=replace(cfg.detector, **_clean(raw["detector"], DetectorConfig, "detector")))
    if "legacy_detector" in raw:
        cfg = replace(cfg, legacy_detector=DetectorConfig(
            **_clean(raw["legacy_detector"], DetectorConfig, "legacy_detector")))
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
        data = _clean(raw["data"], DataConfig, "data")
        if "acceptance" in data:
            data["acceptance"] = _overlay_acceptance(cfg.data.acceptance, data["acceptance"])
        cfg = replace(cfg, data=replace(cfg.data, **data))
    if "engine" in raw:
        cfg = replace(cfg, engine=replace(cfg.engine, **_clean(raw["engine"], EngineConfig, "engine")))

    # n_gamma derived from an inherited scintillator must follow an overridden
    # bar length (it used to keep the base preset's value silently).
    if (base is not None and cfg.sensitivity.scintillator
            and "bar_length_m" in (raw.get("detector") or {})
            and "n_gamma" not in (raw.get("sensitivity") or {})):
        cfg = replace(cfg, sensitivity=replace(
            cfg.sensitivity,
            n_gamma=_scintillator_n_gamma(cfg.sensitivity.scintillator,
                                          cfg.detector.bar_length_m)))

    return cfg


def _overlay_acceptance(base: Dict[str, str], given: Optional[dict]) -> Dict[str, str]:
    """data.acceptance applied per meson: a listed meson replaces the base
    preset's file, a null drops it, and the other mesons keep the base's."""
    given = given or {}
    unknown = sorted(set(given) - set(C.MESONS), key=str)
    if unknown:
        hint = "".join(_did_you_mean(str(m), C.MESONS) for m in unknown)
        raise ValueError(f"unknown meson(s) {unknown} in data.acceptance{hint}; "
                         f"valid: {', '.join(C.MESONS)}")
    merged = {**base, **given}
    return {m: f for m, f in merged.items() if f is not None}


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
        hint = "".join(_did_you_mean(k, allowed) for k in sorted(unknown))
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in the '{section}' config "
            f"section{hint}; valid keys: {sorted(allowed)}"
        )
    return dict(d or {})
