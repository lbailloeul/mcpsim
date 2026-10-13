"""Assemble per-channel mCP yields for a configuration (fast mode).

This is the compute core shared by the production and limit plots and the CLI.
It loads precomputed acceptance / cross-section inputs, evaluates the physics
formulas on a common mass grid, and returns baseline yields (N_chi/eps^2). The
epsilon/detection dependence is applied later in sensitivity.signal_2d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import geometry, paths
from .config import Config, run_config
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
    cfg = run_config(cfg)
    grid_path = paths.resolve_data(cfg.data.mass_grid)
    try:
        grid = np.loadtxt(grid_path).astype(float)
    except FileNotFoundError as exc:
        paths.require_source_repo()   # no data at all: raise the download instructions
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
            result.warnings.append(
                f"upsilon has no acceptance scan: it borrows this run's low-mass "
                f"J/psi acceptance ({finite[0]:.4g}) flat in mass, past the J/psi "
                f"scan's 1.525 GeV end; the channel is ~1e-4 of the J/psi "
                f"(docs/physics.md).")
            return float(finite[0])

    result.warnings.append(
        "no J/psi acceptance was computed for this run, so the upsilon has "
        "nothing to borrow from (it has no scan of its own); skipping the "
        "upsilon channel. Enable the J/psi, or set data.upsilon_ageo."
    )
    return None


def acceptance_files_valid(cfg: Config) -> bool:
    """True when the archived acceptance files (data.acceptance, data.dy_ageo)
    were made for the detector this run models: the preset's own detector, or
    the published one (legacy_detector) when the preset records it."""
    if not cfg.geometry_is_preset:
        return False
    return (cfg.legacy_detector is None
            or geometry.face_signature(cfg.detector) == geometry.face_signature(cfg.legacy_detector))


def _meson_ageo(cfg: Config, name: str, grid: np.ndarray,
                result: ProductionResult):
    """Geometric acceptance for one meson, honest about geometry.

    1. The file was made for this detector (acceptance_files_valid) -> load
       it verbatim (bit-for-bit parity with the published pipeline).
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

    if acceptance_files_valid(cfg) and acc_file:
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

    # The grids also carry the acceptance of the detector they were generated
    # for (single track or shared bar on a 1x1 m face at 40 / 100 m, with the
    # first chi always at azimuth 0, so their theta axis ends at the face's
    # half-width), so brem through any part of the face beyond that angle is
    # underestimated: the corners of the 1x1 m face itself, all of a far
    # off-axis face.
    lim = cfg.data.brem_max_theta_mrad
    if lim is not None:
        reach = 1e3 * geometry.face_max_theta(det)
        if reach > lim * (1 + 1e-6):
            where = ("brem is underestimated (it collapses at low mass)" if det.offaxis_mrad
                     else "brem through the face beyond that angle is missing (underestimated)")
            result.warnings.append(
                f"brem is not accurate above {lim:g} mrad: the grids only resolve "
                f"the face of the detector they were generated for out to that "
                f"angle, and this detector reaches {reach:.3f} mrad, so {where}.")

    # The grids carry sigma_pA of the nucleus they were generated for; the brem
    # readers rescale to this target. Say so, since it moves brem by up to x3.
    files = sorted(brem_dir.glob(cfg.data.brem_pattern))
    if files:
        A_grid = brem_mod.grid_target_A(str(files[0]))
        if A_grid is None:
            result.warnings.append(
                f"brem grids in {brem_dir} do not declare their target "
                f"('# A_<X> = ... g/mol' header); assuming they were generated "
                f"for target '{t.name}' (A={t.A:g}), no sigma_pA rescaling.")
        elif abs(A_grid - t.A) > 1e-6:
            scale = brem_mod.sigma_pA_mb(t.A) / brem_mod.sigma_pA_mb(A_grid)
            result.warnings.append(
                f"brem grids were generated for A={A_grid:g}; rescaled their "
                f"sigma_pA to target '{t.name}' (A={t.A:g}): x{scale:.3f}.")

    if bands[1].sum() == 0:
        result.warnings.append(
            f"no brem files matched {cfg.data.brem_pattern} in {brem_dir}; "
            f"a new beam energy needs --regenerate."
        )
    result.brem_low, result.brem_central, result.brem_high = bands[0], bands[1], bands[2]


def drell_yan_table(cfg: Config) -> Optional[Path]:
    """The built-in per-material Drell-Yan table this run uses, or None.

    Default (fidelity corrected): the table for the configured target and
    beam energy (paths.DY_TABLES_DIR). fidelity = legacy keeps the paper's
    MadGraph scan (data.dy_cross, iron nPDF for both beams), and so does an
    explicit data.dy_cross in a user config."""
    if cfg.engine.fidelity == "legacy" or not cfg.dy_cross_is_preset:
        return None
    return drellyan.find_table(paths.DY_TABLES_DIR, cfg.beam.energy_gev, cfg.target.name)


def _add_drell_yan(cfg: Config, grid: np.ndarray, result: ProductionResult) -> None:
    # Per-nucleon sigma_DY -> per POT: divide by the configured target's
    # sigma_inel/A. The published pipeline used the iron value (13 mb) for
    # every target; legacy fidelity keeps it.
    t = cfg.target
    sigma_mb = (C.SIGMA_PN_INELASTIC_MB if cfg.engine.fidelity == "legacy" else
                drellyan.sigma_inel_per_nucleon_mb(t.density_g_cm3, t.interaction_length_cm))
    table = drell_yan_table(cfg)
    if table is not None:
        tab = drellyan.load_table(table)
        m, y = drellyan.dy_yield_table(tab, cfg.beam.n_pot, sigma_mb)
        face = geometry.face_signature(cfg.detector)
        if tab.acc_detector != face:
            result.warnings.append(
                f"Drell-Yan acceptance is the LO acceptance of the preset's "
                f"detector ({tab.acc_detector}), not yours ({face}); it is not "
                f"recomputed for custom or off-axis detectors. Treat this "
                f"channel as approximate.")
    else:
        if not (cfg.data.dy_cross and cfg.data.dy_ageo):
            result.warnings.append(
                f"no Drell-Yan cross sections: no built-in table for target "
                f"'{t.name}' at {cfg.beam.energy_gev:g} GeV and no data.dy_cross/"
                f"dy_ageo scan; skipping.")
            return
        cross = str(paths.resolve_data(cfg.data.dy_cross))
        ageo = str(paths.resolve_data(cfg.data.dy_ageo))
        try:
            m, y = drellyan.dy_yield(cross, ageo, cfg.beam.n_pot, sigma_inelastic_mb=sigma_mb)
        except OSError:
            result.warnings.append("Drell-Yan input files missing; skipping.")
            return
        _check_dy_scan(cfg, cross, ageo, result)
        if not acceptance_files_valid(cfg):
            result.warnings.append(
                f"Drell-Yan acceptance is the PRESET-geometry file "
                f"({cfg.data.dy_ageo}), not your detector geometry — it is not "
                f"recomputed for custom or off-axis detectors. Treat this channel "
                f"as approximate.")
    order = np.argsort(m)
    result.drell_yan = np.interp(grid, m[order], y[order], left=0.0, right=0.0)


def _check_dy_scan(cfg: Config, cross: str, ageo: str, result: ProductionResult) -> None:
    """Caveats of a MadGraph scan (data.dy_cross) used in place of a table."""
    t, mat = cfg.target, cfg.data.dy_material
    if cfg.engine.fidelity != "legacy" and cfg.dy_cross_is_preset:
        result.warnings.append(
            f"no built-in Drell-Yan table for target '{t.name}' at "
            f"{cfg.beam.energy_gev:g} GeV; using the scan {cfg.data.dy_cross}. "
            f"Build one with `python -m mcpsim.generation.dy_tables` (docs/data.md).")
    if mat is None:
        result.warnings.append(
            f"the Drell-Yan cross sections do not declare the material their "
            f"nuclear PDFs were simulated for (data.dy_material); they are "
            f"applied to target '{t.name}' as they are.")
    elif mat != t.name and cfg.engine.fidelity == "legacy":
        result.warnings.append(
            f"Drell-Yan uses the paper's scan, simulated with {mat} nuclear PDFs, "
            f"and the 13 mb normalisation for target '{t.name}' (fidelity legacy, "
            f"as published).")
    elif mat != t.name:
        result.warnings.append(
            f"Drell-Yan cross sections were simulated with {mat} nuclear PDFs "
            f"and are applied to target '{t.name}': the per-POT normalisation "
            f"follows the target, but the per-nucleon nPDF difference is not "
            f"included (<~10% for m_chi <~ 3 GeV, up to ~25% near the kinematic limit).")
    mismatch = drellyan.row_mismatch(cross, ageo)
    if mismatch:
        result.warnings.append(mismatch)
