"""Assemble per-channel mCP yields for a configuration (fast mode).

This is the compute core shared by the production and limit plots and the CLI.
It loads precomputed acceptance / cross-section inputs, evaluates the physics
formulas on a common mass grid, and returns baseline yields (N_chi/eps^2). The
epsilon/detection dependence is applied later in sensitivity.signal_2d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import geometry, paths
from .config import Config
from .physics import brem as brem_mod
from .physics import constants as C
from .physics import drellyan, meson_decay


@dataclass
class ProductionResult:
    """All computed channels on the config's mass grid.

    Yields are baseline N_chi/eps^2 (the eps dependence is applied later by
    physics/sensitivity). meson_channels maps meson name -> yield array;
    brem_low/central/high are the Lambda_p = 1.0/1.5/2.0 GeV band (central
    enters `total`). `warnings` carries every degradation/skip message and is
    printed by the CLI at the end of the run."""
    masses: np.ndarray
    meson_channels: Dict[str, np.ndarray] = field(default_factory=dict)
    brem_central: np.ndarray = None
    brem_low: np.ndarray = None
    brem_high: np.ndarray = None
    drell_yan: np.ndarray = None
    warnings: List[str] = field(default_factory=list)

    @property
    def meson_total(self) -> np.ndarray:
        if not self.meson_channels:
            return np.zeros_like(self.masses)
        return np.sum(list(self.meson_channels.values()), axis=0)

    @property
    def total(self) -> np.ndarray:
        tot = self.meson_total.copy()
        if self.drell_yan is not None:
            tot = tot + self.drell_yan
        if self.brem_central is not None:
            tot = tot + self.brem_central
        return tot


def _load_acceptance_on_grid(path, grid: np.ndarray) -> np.ndarray:
    """Load a (mass, acceptance) table and interpolate onto `grid` (zero-fill).

    Tolerant of a text header row and extra columns: the LANL efficiency
    summaries carry a `mass_GeV eff hits total ...` header and 11 columns, while
    the DarkQuest/SHiP tables are bare 2-column files. Column 0 is mass, column 1
    is the geometric acceptance.
    """
    arr = np.genfromtxt(path, comments="#")
    if arr.ndim == 1:
        # Ambiguous 1-D data: exactly two values reads as one (mass, ageo)
        # row; anything else is a single column of acceptances. (atleast_2d
        # alone would flatten an n-row single-column file into one row.)
        arr = arr[None, :] if arr.size == 2 else arr[:, None]
    arr = arr[~np.isnan(arr).any(axis=1)]  # drop a non-numeric header row
    if arr.size == 0:
        return np.zeros_like(grid)
    if arr.shape[1] == 1:                  # single column -> assume aligned to grid
        if arr.shape[0] < grid.size:
            raise ValueError(
                f"single-column acceptance file {path} has {arr.shape[0]} rows "
                f"but the mass grid has {grid.size} points; the rows cannot be "
                f"aligned. Regenerate the file or add a mass column."
            )
        return arr[: grid.size, 0]
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order], left=0.0, right=0.0)


def _active_channels(cfg: Config) -> tuple[List[str], List[str]]:
    """Filter requested channels, disabling brem/DY in collider mode."""
    warnings: List[str] = []
    channels = list(cfg.channels)
    if cfg.beam.is_collider:
        for ch in ("brem", "drell_yan"):
            if ch in channels:
                channels.remove(ch)
                warnings.append(
                    f"channel '{ch}' disabled: the bremsstrahlung/Drell-Yan "
                    f"machinery is fixed-target only (collider frame requested)."
                )
    return channels, warnings


def compute_production(cfg: Config) -> ProductionResult:
    """Evaluate all configured channels on the config's mass grid."""
    grid_path = paths.resolve_data(cfg.data.mass_grid)
    try:
        grid = np.loadtxt(grid_path).astype(float)
    except FileNotFoundError as exc:
        if not paths.source_repo().exists():
            raise FileNotFoundError(
                f"mass grid '{cfg.data.mass_grid}' not found (tried {grid_path}) "
                f"because the data source repo is missing. Set MCPSIM_SOURCE_REPO "
                f"or the 'source_repo' config field to your github-repo-scripts "
                f"checkout / data bundle."
            ) from exc
        raise FileNotFoundError(
            f"mass grid '{cfg.data.mass_grid}' not found (tried {grid_path}). "
            f"Check the data.mass_grid config value against the source repo at "
            f"{paths.source_repo()}."
        ) from exc
    grid = np.atleast_1d(grid)
    channels, warnings = _active_channels(cfg)
    result = ProductionResult(masses=grid, warnings=list(cfg.notes) + warnings)

    if "meson_decay" in channels:
        _add_meson_channels(cfg, grid, result)
    if "brem" in channels:
        _add_brem(cfg, grid, result)
    if "drell_yan" in channels:
        _add_drell_yan(cfg, grid, result)

    if not np.any(result.total > 0):
        result.warnings.append(
            "no channel produced any yield — the plots will be empty. Check the "
            "acceptance/brem/DY data paths for this config, or run --regenerate."
        )
    return result


def _add_meson_channels(cfg: Config, grid: np.ndarray, result: ProductionResult) -> None:
    rates = cfg.c_meson_rates()
    computed: Dict[str, np.ndarray] = {}
    for name, spec in C.MESONS.items():
        if name not in rates:
            continue  # not produced at this beam energy (e.g. heavy states at low E)
        c = rates[name]

        if name == "upsilon":
            # MESONS orders jpsi before upsilon, so its acceptance for THIS run
            # is already in hand.
            ua = _upsilon_ageo(cfg, computed.get("jpsi"), result)
            if ua is None:
                continue
            ageo = np.full(grid.size, ua)
        else:
            ageo = _meson_ageo(cfg, name, grid, result)
            if ageo is None:
                continue

        computed[name] = ageo
        result.meson_channels[name] = meson_decay.meson_yield(
            spec, grid, ageo, cfg.beam.n_pot, c, fidelity=cfg.engine.fidelity
        )


def _upsilon_ageo(cfg: Config, jpsi_ageo, result: ProductionResult):
    """Flat acceptance for the upsilon, which has no scan of its own.

    Preference order:
      1. an explicit data.upsilon_ageo,
      2. the low-mass value of the J/psi acceptance *as computed for this run*,
         so the borrowed number always matches the geometry actually used.
    `family` gates inclusion (C.UPSILON_FAMILIES).
    Returns None (with a warning) when the family is unknown or no J/psi
    acceptance exists; the channel is then skipped. See docs/physics.md for
    why the borrow is needed at all.
    """
    if cfg.data.upsilon_ageo is not None:
        return cfg.data.upsilon_ageo

    # `family` decides whether the channel is included at all: an unrecognised
    # family has no basis for the borrow and skips, exactly as before.
    if cfg.family not in C.UPSILON_FAMILIES:
        result.warnings.append(
            f"no upsilon acceptance for family '{cfg.family}' (no scan file "
            f"exists in the legacy data; the known families are "
            f"darkquest/ship). Set data.upsilon_ageo to include it; "
            f"skipping the upsilon channel."
        )
        return None

    # The value is always this run's own J/psi acceptance, so it matches the
    # geometry actually computed. Without one there is nothing to borrow from
    # and the channel is skipped rather than falling back on a number measured
    # at some other detector.
    if jpsi_ageo is not None:
        finite = jpsi_ageo[np.isfinite(jpsi_ageo) & (jpsi_ageo > 0)]
        if finite.size:
            return float(finite[0])

    result.warnings.append(
        "no J/psi acceptance was computed for this run, so the upsilon has "
        "nothing to borrow from (it has no scan of its own); skipping the "
        "upsilon channel. Enable the J/psi, or set data.upsilon_ageo."
    )
    return None


def _meson_ageo(cfg: Config, name: str, grid: np.ndarray,
                result: ProductionResult):
    """Geometric acceptance for one meson, honest about geometry.

    1. Preset geometry + a configured file -> load the file verbatim
       (bit-for-bit parity with the published pipeline).
    2. Overridden geometry + the Python engine + archived PYTHIA samples ->
       recompute for the ACTUAL geometry (cached under mcpsim-output/cache).
    3. Otherwise fall back to the preset file WITH A LOUD WARNING — the file
       was computed for the preset's detector, not this one. (Previously this
       fallback happened silently, so e.g. a 60 m custom detector reused the
       40 m DarkQuest acceptance with no indication.)
    """
    acc_file = cfg.data.acceptance.get(name)

    def _load_file():
        try:
            return _load_acceptance_on_grid(paths.resolve_data(acc_file), grid)
        except OSError:
            result.warnings.append(
                f"acceptance file missing for '{name}' ({acc_file}); skipping.")
            return None

    if cfg.geometry_is_preset and acc_file:
        return _load_file()

    if cfg.engine.decay == "python":
        from .generation import pydecay, samples
        if samples.have_samples(cfg, name):
            return pydecay.cached_acceptance(cfg, name, grid,
                                             paths.CACHE_DIR / "acceptance")
        reason = (f"archived PYTHIA sample for '{name}' not found under "
                  f"{samples.samples_dir(cfg)}")
    else:
        reason = ("engine.decay = 'cpp': run `mcpsim generate mesons --engine cpp` "
                  "to produce acceptance for this geometry")

    if acc_file:
        result.warnings.append(
            f"'{name}' acceptance uses the PRESET-geometry file ({acc_file}), "
            f"not your detector geometry — {reason}. Treat this channel as "
            f"approximate."
        )
        return _load_file()
    result.warnings.append(f"no acceptance available for '{name}' ({reason}); skipping.")
    return None


def _add_brem(cfg: Config, grid: np.ndarray, result: ProductionResult) -> None:
    if not cfg.data.brem_dir:
        result.warnings.append("no brem_dir configured; skipping bremsstrahlung.")
        return
    brem_dir = paths.resolve_data(cfg.data.brem_dir)
    t = cfg.target

    # Off-axis faces (and bar arrays under corrected fidelity) use the exact
    # azimuthal kernel; the on-axis cone cut / equal-area approximation stays
    # as the legacy-parity path.
    det = cfg.detector
    exact_kernel = (det.offaxis_mrad != 0.0
                    or (det.type == "bar_array"
                        and cfg.engine.fidelity == "corrected"))

    bands = {}
    for idx in (0, 1, 2):
        if exact_kernel:
            m, y = brem_mod.brem_yield_offaxis(
                str(brem_dir), cfg.data.brem_pattern, det,
                cfg.beam.n_pot, t.density_g_cm3, t.interaction_length_cm, t.A,
                lambda_idx=idx,
            )
        else:
            det_angle = geometry.acceptance_theta_cut(det)
            m, y = brem_mod.brem_yield(
                str(brem_dir), cfg.data.brem_pattern, det_angle,
                cfg.beam.n_pot, t.density_g_cm3, t.interaction_length_cm, t.A,
                lambda_idx=idx,
            )
        bands[idx] = brem_mod.interp_to_grid(m, y, grid)

    if bands[1].sum() == 0:
        result.warnings.append(
            f"no brem files matched {cfg.data.brem_pattern} in {brem_dir}; "
            f"a new beam energy needs --regenerate."
        )
    result.brem_low, result.brem_central, result.brem_high = bands[0], bands[1], bands[2]


def _add_drell_yan(cfg: Config, grid: np.ndarray, result: ProductionResult) -> None:
    if not (cfg.data.dy_cross and cfg.data.dy_ageo):
        result.warnings.append("no Drell-Yan inputs configured; skipping (use --regenerate to run MadGraph).")
        return
    try:
        m, y = drellyan.dy_yield(
            str(paths.resolve_data(cfg.data.dy_cross)),
            str(paths.resolve_data(cfg.data.dy_ageo)),
            cfg.beam.n_pot,
        )
    except OSError:
        result.warnings.append("Drell-Yan input files missing; skipping.")
        return
    mismatch = drellyan.row_mismatch(str(paths.resolve_data(cfg.data.dy_cross)),
                                     str(paths.resolve_data(cfg.data.dy_ageo)))
    if mismatch:
        result.warnings.append(mismatch)
    order = np.argsort(m)
    result.drell_yan = np.interp(grid, m[order], y[order], left=0.0, right=0.0)
