"""End-to-end orchestration: (optionally regenerate) -> compute -> plot.

Fast mode evaluates the physics on the precomputed inputs. Regenerate mode runs
the heavy backends per channel, wiring their outputs back into the config; each
channel degrades gracefully to the precomputed input if its toolchain is absent.
"""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import geometry, paths
from .config import Config, mark_acceptance_supplied, run_config, run_record
from .model import ProductionResult, compute_production, drell_yan_table, load_mass_grid
from .physics.constants import MESONS
from .plotting.limits import plot_limit
from .plotting.production import plot_production


def run_pipeline(cfg: Config, *, regenerate: bool = False,
                 out_dir: Optional[Path] = None,
                 mass_subset: Optional[int] = None,
                 gen_opts: Optional[dict] = None,
                 plots: Sequence[str] = ("production", "limit"),
                 ) -> Tuple[ProductionResult, List[Path]]:
    """Run the full pipeline; return (result, [written plot + table paths]).

    `plots` selects which figures to draw; the numeric tables and the run
    record are always written."""
    cfg = run_config(cfg)
    tag = cfg.tag
    out_dir = paths.output_dir(out_dir, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_warnings: List[str] = []
    if regenerate:
        cfg, gen_warnings = regenerate_inputs(cfg, out_dir / "generated", mass_subset, gen_opts)

    result = compute_production(cfg)
    result.warnings = gen_warnings + result.warnings

    written: List[Path] = []
    if "production" in plots:
        written.append(plot_production(result, out_dir / f"production_{tag}.pdf"))
    if "limit" in plots:
        written.append(plot_limit(result, out_dir / f"limit_{tag}.pdf"))
    return result, written + write_tables(result, out_dir)


def write_tables(result: ProductionResult, out_dir: Path) -> List[Path]:
    """Write the numeric results next to the plots.

    production_<tag>.txt: mass plus the baseline yield (N_chi/eps^2) per
    channel and the total. limit_<tag>.txt: mass and the epsilon at which the
    signal reaches the config's threshold (NaN where it never does). These
    tables are also the golden-snapshot format for the legacy-parity tests.
    """
    from .physics import sensitivity

    cfg = result.config
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
    eps95 = sensitivity.exclusion_contour(result.total, s)
    floor = sensitivity.charge_grid(s)[0]
    lim_path = out_dir / f"limit_{tag}.txt"
    np.savetxt(lim_path, np.column_stack([result.masses, eps95]), fmt="%.8g",
               header=f"mass_GeV eps_at_{s.n_chi_threshold}_events "
               f"[D(eps)=(1-exp(-{s.n_gamma:g} eps^2))^{s.a:g} eps^2]")
    # exclusion_contour returns charge_min where the signal already passes the
    # threshold at the bottom of the scan; that is a floor, not a limit.
    clipped = int(np.sum(eps95 == floor))
    if clipped:
        msg = (f"the limit hits the bottom of the charge scan (charge_min = "
               f"{s.charge_min:g}) at {clipped} mass point(s): the true limit is "
               f"stronger there. Lower sensitivity.charge_min.")
        if msg not in result.warnings:
            result.warnings.append(msg)
    curve_path = write_limit_curve(result, eps95, floor, out_dir)
    return [prod_path, lim_path, curve_path, write_run_record(cfg, out_dir)]


def write_limit_curve(result: ProductionResult, eps: np.ndarray, floor: float,
                      out_dir: Path) -> Path:
    """limit_curve_<tag>.txt: the exclusion boundary alone, for collaborators to
    replot — the masses where the threshold is reached, with a header that
    says what the curve is. Whitespace columns with '#' comments, so it loads
    with np.loadtxt, ROOT's TGraph or gnuplot as it is."""
    from . import __version__, geometry
    cfg, masses = result.config, result.masses
    s, b = cfg.sensitivity, cfg.beam
    ok = np.isfinite(eps)
    n_floor = int(np.sum(ok & (eps <= floor)))
    # A requested channel can be skipped at run time (e.g. brem past its grids).
    gave_yield = {"meson_decay": bool(result.meson_channels),
                  "brem": result.brem_central is not None,
                  "drell_yan": result.drell_yan is not None}
    channels = ", ".join(c for c in cfg.channels if gave_yield[c]) or "none"
    lines = [
        f"mcpsim limit curve: {cfg.name}",
        f"The excluded region lies above this curve: the epsilon = Q/e at which the "
        f"expected signal reaches N = {s.n_chi_threshold:g} events",
        "(3.09 = 95% CL for zero background, Feldman-Cousins; see docs/physics.md).",
        f"Signal: N(m, eps) = Y(m) * eps^2 * (1 - exp(-n_gamma eps^2))^a, "
        f"n_gamma = {s.n_gamma:g} photoelectrons at eps = 1, a = {s.a:g}",
        f"Beam: {b.energy_gev:g} GeV protons on {cfg.target.name}, {b.n_pot:.3g} POT ({b.frame})",
        f"Detector: {geometry.face_signature(cfg.detector)}",
        f"Channels: {channels} (fidelity {cfg.engine.fidelity})",
        f"{int(np.sum(~ok))} of {masses.size} mass points never reach N and are omitted.",
    ]
    if n_floor:
        lines.append(f"At {n_floor} mass point(s) the signal already passes N at the "
                     f"bottom of the charge scan (eps = {floor:g}): a floor, the true "
                     f"limit is stronger there.")
    lines += [f"Full configuration: run_{cfg.tag}.yaml; per-channel yields: "
              f"production_{cfg.tag}.txt",
              f"mcpsim {__version__}, {datetime.date.today().isoformat()}",
              "m_chi [GeV]    epsilon"]
    path = Path(out_dir) / f"limit_curve_{cfg.tag}.txt"
    np.savetxt(path, np.column_stack([masses[ok], eps[ok]]), fmt="%.6e",
               header="\n".join(lines))
    return path


def write_run_record(cfg: Config, out_dir: Path) -> Path:
    """Record the fully resolved configuration (preset + config + CLI flags)
    next to the outputs, so every table and plot can be traced, and rerun
    with `mcpsim run --config run_<tag>.yaml`."""
    import yaml

    dy = None
    if "drell_yan" in cfg.channels:
        table = drell_yan_table(cfg)
        dy = str(table) if table else cfg.data.dy_cross
    record = run_record(cfg, data_source=str(paths.source_repo()), drell_yan_cross_sections=dy)
    path = Path(out_dir) / f"run_{cfg.tag}.yaml"
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def regenerate_inputs(cfg: Config, gen_dir: Path, mass_subset: Optional[int] = None,
                      gen_opts: Optional[dict] = None) -> Tuple[Config, List[str]]:
    """Run heavy backends; return (config pointing at fresh outputs, warnings).

    Imported lazily so fast mode never needs the generation dependencies.
    `gen_opts` tunes resolution: trials, n_proc, brem_grid_size, brem_neval,
    brem_nitn, dy_nevents (all optional; one left out keeps the backend's own
    default), and condor/submit send brem and Drell-Yan to HTCondor batches.
    Every acceptance file it writes is recorded as made for cfg.detector, the
    detector the backends were given. Each channel that cannot run (missing
    toolchain, unsupported geometry) degrades to its precomputed input; the
    reason is printed immediately AND returned so the CLI can repeat it in the
    end-of-run report instead of letting it scroll away mid-run.
    """
    from .generation import brem_run, decay, madgraph, mesons
    from .generation.common import ToolchainError

    warnings: List[str] = []

    def _skip(msg: str) -> None:
        print(f"[mcpsim] {msg}")
        warnings.append(msg)

    opts = gen_opts or {}
    gen_dir.mkdir(parents=True, exist_ok=True)
    grid = load_mass_grid(cfg)
    if mass_subset:
        grid = grid[:: max(1, grid.size // mass_subset)][:mass_subset]

    data = cfg.data
    written: List[str] = []

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
                continue
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
                written.append(acc[name])
            except ToolchainError as exc:
                _skip(f"meson '{name}' regeneration skipped: {exc}")
        data = replace(data, acceptance=acc)

    if "brem" in cfg.channels:
        try:
            brem_masses = grid[grid < 1.0]
            brem_kw = _given(opts, n_proc="n_proc", brem_nitn="nitn",
                             brem_neval="neval", brem_grid_size="grid_size")
            if opts.get("condor"):
                # Emit/submit an HTCondor batch; jobs run asynchronously, so the
                # brem files won't exist yet this run.
                out = brem_run.prepare_condor_brem(cfg, brem_masses, gen_dir / "brem",
                                                   **brem_kw, **_given(opts, submit="submit"))
                warnings.append(
                    f"brem is running as an HTCondor batch in {gen_dir / 'brem'}; "
                    f"this run has none until the jobs finish (see README_condor.txt there).")
            else:
                out = brem_run.regenerate_brem(cfg, brem_masses, gen_dir / "brem", **brem_kw)
            data = replace(data, brem_dir=str(out), brem_pattern="Brem_*.txt")
        except ToolchainError as exc:
            _skip(f"brem regeneration skipped (using precomputed): {exc}")

    dy_scan = False
    if "drell_yan" in cfg.channels:
        try:
            dy_kw = _given(opts, n_proc="n_proc", dy_nevents="nevents")
            if opts.get("condor"):
                dy_cross, dy_ageo = madgraph.prepare_condor_dy(
                    cfg, grid, gen_dir / "dy", **dy_kw, **_given(opts, submit="submit"))
                warnings.append(
                    f"Drell-Yan is running as an HTCondor batch in {gen_dir / 'dy'}; "
                    f"this run has none until the jobs finish (see README_condor.txt there).")
            else:
                dy_cross, dy_ageo = madgraph.generate_drell_yan(cfg, grid, gen_dir / "dy",
                                                                **dy_kw)
            # MadGraph runs with the archived iron nPDF for both beams. The
            # scan is used as it is, like a data.dy_cross from a config, not
            # replaced by the built-in table (model.drell_yan_table).
            data = replace(data, dy_cross=str(dy_cross), dy_ageo=str(dy_ageo),
                           dy_material=madgraph.DY_PDF_MATERIAL)
            written.append(str(dy_ageo))
            dy_scan = True
        except ToolchainError as exc:
            _skip(f"Drell-Yan regeneration skipped (using precomputed): {exc}")

    cfg = mark_acceptance_supplied(replace(cfg, data=data), written,
                                   geometry.face_signature(cfg.detector))
    return (replace(cfg, dy_cross_is_preset=False) if dy_scan else cfg), warnings


def _trials(mass_subset: Optional[int]) -> int:
    return 100_000 if mass_subset else 10_000_000


def _given(opts: dict, **names: str) -> dict:
    """Backend keyword arguments for the gen_opts the caller set, renamed from
    gen_opts key to backend parameter. An option left out is not forwarded,
    so the backend signature is the one place its default lives."""
    return {param: opts[key] for key, param in names.items() if opts.get(key) is not None}
