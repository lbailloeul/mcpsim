"""End-to-end orchestration: (optionally regenerate) -> compute -> plot.

Fast mode evaluates the physics on the precomputed inputs. Regenerate mode runs
the heavy backends per channel, wiring their outputs back into the config; each
channel degrades gracefully to the precomputed input if its toolchain is absent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import paths
from .config import Config
from .model import ProductionResult, compute_production
from .physics.constants import MESONS
from .plotting.limits import plot_limit
from .plotting.production import plot_production


def run_pipeline(cfg: Config, *, regenerate: bool = False,
                 channels: Optional[List[str]] = None,
                 out_dir: Optional[Path] = None,
                 mass_subset: Optional[int] = None,
                 gen_opts: Optional[dict] = None,
                 make_plots: bool = True) -> Tuple[ProductionResult, List[Path]]:
    """Run the full pipeline; return (result, [plot paths])."""
    if channels:
        cfg = replace(cfg, channels=list(channels))
    tag = cfg.tag
    # Resolve to absolute so generated-data paths survive resolve_data (which
    # otherwise interprets relative paths against the source repo).
    out_dir = Path(out_dir).resolve() if out_dir else (paths.WORK_DIR / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_warnings: List[str] = []
    if regenerate:
        cfg, gen_warnings = regenerate_inputs(cfg, out_dir / "generated", mass_subset, gen_opts)

    result = compute_production(cfg)
    result.warnings = gen_warnings + result.warnings
    write_tables(cfg, result, out_dir)

    plots: List[Path] = []
    if make_plots:
        plots.append(plot_production(cfg, result, out_dir / f"production_{tag}.pdf"))
        plots.append(plot_limit(cfg, result, out_dir / f"limit_{tag}.pdf"))
    return result, plots


def write_tables(cfg: Config, result: ProductionResult, out_dir: Path) -> List[Path]:
    """Write the numeric results next to the plots.

    production_<tag>.txt: mass plus the baseline yield (N_chi/eps^2) per
    channel and the total. limit_<tag>.txt: mass and the epsilon at which the
    signal reaches the config's threshold (NaN where it never does). These
    tables are also the golden-snapshot format for the legacy-parity tests.
    """
    from .physics import sensitivity

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag

    cols: List[np.ndarray] = [result.masses]
    names: List[str] = ["mass_GeV"]
    for meson, y in result.meson_channels.items():
        cols.append(y)
        names.append(meson)
    for label, y in (("brem_low", result.brem_low), ("brem", result.brem_central),
                     ("brem_high", result.brem_high), ("drell_yan", result.drell_yan)):
        if y is not None:
            cols.append(y)
            names.append(label)
    cols.append(result.total)
    names.append("total")
    prod_path = out_dir / f"production_{tag}.txt"
    np.savetxt(prod_path, np.column_stack(cols), fmt="%.8g",
               header=" ".join(names) + "   [baseline N_chi/eps^2, "
               f"{cfg.beam.n_pot:g} POT]")

    s = cfg.sensitivity
    charges = sensitivity.charge_grid(s.charge_min, s.charge_max, s.n_charge)
    eps95 = sensitivity.exclusion_contour(result.total, charges, s.n_gamma, s.a,
                                          s.n_chi_threshold)
    lim_path = out_dir / f"limit_{tag}.txt"
    np.savetxt(lim_path, np.column_stack([result.masses, eps95]), fmt="%.8g",
               header=f"mass_GeV eps_at_{s.n_chi_threshold}_events "
               f"[D(eps)=(1-exp(-{s.n_gamma:g} eps^2))^{s.a:g} eps^2]")
    return [prod_path, lim_path]


def regenerate_inputs(cfg: Config, gen_dir: Path, mass_subset: Optional[int] = None,
                      gen_opts: Optional[dict] = None) -> Tuple[Config, List[str]]:
    """Run heavy backends; return (config pointing at fresh outputs, warnings).

    Imported lazily so fast mode never needs the generation dependencies.
    `gen_opts` tunes resolution: trials, n_proc, brem_grid_size, brem_neval,
    brem_nitn (all optional; sensible defaults otherwise). Each channel that
    cannot run (missing toolchain, unsupported geometry) degrades to its
    precomputed input; the reason is printed immediately AND returned so the
    CLI can repeat it in the end-of-run report instead of letting it scroll
    away mid-run.
    """
    from .generation import brem_run, decay, madgraph, mesons
    from .generation.common import ToolchainError

    warnings: List[str] = []

    def _skip(msg: str) -> None:
        print(f"[mcpsim] {msg}")
        warnings.append(msg)

    opts = gen_opts or {}
    gen_dir.mkdir(parents=True, exist_ok=True)
    grid = np.atleast_1d(np.loadtxt(paths.resolve_data(cfg.data.mass_grid)).astype(float))
    if mass_subset:
        grid = grid[:: max(1, grid.size // mass_subset)][:mass_subset]

    data = cfg.data

    if "meson_decay" in cfg.channels:
        # Yields need per-POT multiplicities; regeneration cannot supply them
        # (the meson backends produce kinematics, not rates). Fail BEFORE the
        # expensive acceptance work, not after.
        try:
            cfg.c_meson_rates()
        except ValueError as exc:
            _skip(f"meson_decay regeneration skipped: {exc}")
            cfg = replace(cfg, channels=[c for c in cfg.channels if c != "meson_decay"])

    if "meson_decay" in cfg.channels:
        acc = dict(data.acceptance)
        for name in MESONS:
            if name == "upsilon":
                continue  # no archived sample and negligible rate; flat default
            try:
                # Preflight BEFORE the (expensive) meson generation: a
                # 10M-trial PYTHIA run must not precede an "unsupported
                # detector" error.
                decay.check_supported(cfg, name)
                if cfg.engine.decay == "python":
                    from .generation import pydecay
                    out = gen_dir / "decay" / f"acceptance_{name}.txt"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    ag = pydecay.acceptance_on_grid(cfg, name, grid)
                    np.savetxt(out, np.column_stack([grid, ag]), fmt="%.8g",
                               header="mass_GeV ageo(per-chi hit prob) [pydecay]")
                    acc[name] = str(out)
                else:
                    source = mesons.generate(cfg, name, gen_dir / "mesons",
                                             trials=opts.get("trials", _trials(mass_subset)))
                    acc[name] = str(decay.run_decay_scan(cfg, name, source, grid,
                                                         gen_dir / "decay"))
            except ToolchainError as exc:
                _skip(f"meson '{name}' regeneration skipped: {exc}")
        data = replace(data, acceptance=acc)

    if "brem" in cfg.channels:
        try:
            brem_masses = grid[grid < 1.0]
            if opts.get("condor"):
                # Emit/submit an HTCondor batch; jobs run asynchronously, so the
                # brem files won't exist yet this run.
                out = brem_run.prepare_condor_brem(
                    cfg, brem_masses, gen_dir / "brem",
                    n_proc=opts.get("n_proc", 8), nitn=opts.get("brem_nitn", 10),
                    neval=opts.get("brem_neval", 4000),
                    grid_size=opts.get("brem_grid_size", 100),
                    submit=opts.get("submit", True),
                )
            else:
                out = brem_run.regenerate_brem(
                    cfg, brem_masses, gen_dir / "brem",
                    n_proc=opts.get("n_proc", 4), nitn=opts.get("brem_nitn", 10),
                    neval=opts.get("brem_neval", 4000),
                    grid_size=opts.get("brem_grid_size", 100),
                )
            data = replace(data, brem_dir=str(out), brem_pattern="Brem_*.txt")
        except ToolchainError as exc:
            _skip(f"brem regeneration skipped (using precomputed): {exc}")

    if "drell_yan" in cfg.channels:
        try:
            dy_cross, dy_ageo = madgraph.generate_drell_yan(cfg, grid, gen_dir / "dy")
            data = replace(data, dy_cross=str(dy_cross), dy_ageo=str(dy_ageo))
        except ToolchainError as exc:
            _skip(f"Drell-Yan regeneration skipped (using precomputed): {exc}")

    return replace(cfg, data=data), warnings


def _trials(mass_subset: Optional[int]) -> int:
    return 100_000 if mass_subset else 10_000_000
