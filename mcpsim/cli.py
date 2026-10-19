"""mcpsim command-line interface.

    mcpsim presets                       list built-in experiment presets
    mcpsim doctor [--check-data]         environment + data status
    mcpsim run --preset darkquest        production + limit plots (fast mode)
    mcpsim run --config my.yaml          custom beam/target/detector
    mcpsim run --config my.yaml --regenerate   full PYTHIA/VEGAS/MadGraph
    mcpsim production / mcpsim limit      a single plot
    mcpsim generate mesons|brem|dy ...   one heavy generation step
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

import click

from . import paths
from .config import (Config, list_presets, load_config, load_preset, normalize_channels,
                     normalize_target, pin_supplied_acceptance, run_config)


def _resolve(preset: Optional[str], config: Optional[str]) -> Config:
    """Load the config and verify the data source repo before any compute."""
    if preset and config:
        raise click.UsageError(
            "give either --preset or --config, not both (a config chooses its "
            "base preset with a `preset:` key).")
    if config:
        if not Path(config).exists():
            raise click.UsageError(f"config file not found: {config}")
        try:
            cfg = load_config(config)
        except ValueError as exc:
            raise click.ClickException(f"in {config}: {exc}")
    elif preset:
        try:
            cfg = load_preset(preset)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc))
    else:
        raise click.UsageError("provide either --preset NAME or --config PATH.")
    if cfg.needs_source_repo:
        try:
            paths.require_source_repo()
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc))
    return cfg


class _clean_errors:
    """Context manager turning compute-time config/data errors into clean CLI
    errors (a beam energy without tabulated multiplicities, a missing data
    file, ...) instead of raw tracebacks."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(exc_type, (ValueError, FileNotFoundError)):
            raise click.ClickException(str(exc))
        return False


def _options(*opts):
    """Decorator applying click options in the order listed."""
    def deco(func):
        for opt in reversed(opts):
            func = opt(func)
        return func
    return deco


_common = [
    click.option("--preset", "-p", default=None, help="Built-in preset name (see `mcpsim presets`)."),
    click.option("--config", "-c", default=None, help="Path to a YAML config."),
    click.option("--out", "-o", "out", default=None,
                 help="Output directory [default: mcpsim-output/<experiment-tag>/]."),
]

_channels_opt = click.option("--channels", default=None,
                             help="Comma-separated subset of: meson_decay,brem,drell_yan "
                                  "(aliases: mesons, dy).")

_offaxis_opt = click.option("--offaxis", default=None,
                            help="Off-axis angle(s) in mrad, comma-separated (e.g. '4,6.5,9'); "
                                 "one run per angle, plus overlay plots for several.")

_engine_opts = [
    click.option("--engine", type=click.Choice(["python", "cpp"]), default=None,
                 help="Decay-acceptance engine [default: config engine.decay]."),
    click.option("--fidelity", type=click.Choice(["corrected", "legacy"]), default=None,
                 help="Physics fidelity: corrected samplers vs bit-for-bit legacy."),
    click.option("--seed", type=int, default=None, help="Engine RNG seed."),
    click.option("--quick", is_flag=True,
                 help="Cap sample statistics for a fast smoke run (not cached)."),
    click.option("--ngamma", type=float, default=None,
                 help="Override sensitivity.n_gamma (photoelectrons at eps = 1); "
                      "e.g. compare light-yield assumptions without editing the config."),
    click.option("--target", default=None,
                 help="Swap the target material: iron (DarkQuest), molybdenum (SHiP), "
                      "tungsten, carbon (graphite); aliases fe, mo, tzm, w, c, graphite. "
                      "Changes brem and the Drell-Yan cross sections (per-material "
                      "tables); meson rates are per POT."),
]


def _apply_engine(cfg: Config, engine, fidelity, seed, quick, ngamma=None,
                  target=None) -> Config:
    """Overlay CLI engine/sensitivity/target flags onto the config."""
    if target is not None:
        try:
            name = normalize_target(target)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--target")
        if name != cfg.target.name:
            # Own name -> own output directory (darkquest_molybdenum/), so a
            # material swap never overwrites the preset's run.
            cfg = replace(cfg, name=f"{cfg.name} {name}",
                          target=cfg.target.overlay({"name": name}))
    eng = cfg.engine
    if engine is not None:
        eng = replace(eng, decay=engine)
    if fidelity is not None:
        eng = replace(eng, fidelity=fidelity)
    if seed is not None:
        eng = replace(eng, seed=seed)
    if quick:
        eng = replace(eng, quick=True)
    if eng is not cfg.engine:
        cfg = replace(cfg, engine=eng)
    if ngamma is not None:
        cfg = replace(cfg, sensitivity=replace(cfg.sensitivity, n_gamma=ngamma))
    if cfg.engine.fidelity == "legacy" and "legacy" not in cfg.name.lower():
        # own name -> own output directory (darkquest_legacy/), so a legacy run
        # never overwrites the default one, and its plots say what they are
        cfg = replace(cfg, name=f"{cfg.name} legacy")
    return run_config(cfg)        # legacy on a preset -> the published detector


def _with_offaxis(cfg: Config, angle_mrad: float) -> Config:
    """Config displaced to one explicit off-axis angle.

    Always renames (so scan outputs land in per-angle directories, even for
    the angle matching the config default) and recomputes acceptance — an
    explicit --offaxis request opts into the off-axis machinery. Files the
    config names are pinned to the detector before the move, so a moved
    detector recomputes them too, and says so."""
    return replace(
        pin_supplied_acceptance(cfg),
        name=f"{cfg.name} {angle_mrad:g}mrad",
        detector=replace(cfg.detector, offaxis_mrad=angle_mrad),
        geometry_is_preset=False,
    )


@click.group()
@click.version_option(package_name="mcpsim")
def cli() -> None:
    """Millicharged-particle production & sensitivity CLI."""


@cli.command("install")
@click.option("--extras", default="gen,dev",
              help="pip extras to install (gen=mpi4py+vegas, dev=pytest).")
@click.option("--dry-run", is_flag=True, help="Show actions without installing.")
@click.option("--check-data", is_flag=True,
              help="Verify the resolved source repo against data/MANIFEST.yaml "
                   "(sha256) instead of installing.")
def install_cmd(extras: str, dry_run: bool, check_data: bool) -> None:
    """Install Python dependencies and report the environment status (doctor).

    pip installs what it can (mpi4py, vegas, pytest). PYTHIA, ROOT and
    MadGraph are external toolchains pip cannot provide; their availability is
    reported, with a scan of conda environments for convenience. Also reports
    how the data source repo resolves — the most common first-run problem.
    """
    import subprocess
    import sys

    if check_data:
        raise SystemExit(_check_data())

    if (paths.PROJECT_ROOT / "pyproject.toml").exists():
        spec = f"{paths.PROJECT_ROOT}[{extras}]"
        cmd = [sys.executable, "-m", "pip", "install", "-e", spec]
        click.echo(f"pip: {' '.join(cmd)}")
        if not dry_run:
            rc = subprocess.run(cmd).returncode
            click.secho(f"pip install {'ok' if rc == 0 else 'FAILED (exit %d)' % rc}",
                        fg="green" if rc == 0 else "red")
    else:
        click.secho("not a source checkout (no pyproject.toml next to the package); "
                    "skipping pip install — run `pip install 'mcpsim[gen,dev]'` "
                    "or install from a git checkout instead.", fg="yellow")
    _doctor_report()


@cli.command("doctor")
@click.option("--check-data", is_flag=True,
              help="Also verify every data file against data/MANIFEST.yaml (sha256).")
def doctor_cmd(check_data: bool) -> None:
    """Report what this machine can run: Python modules, where the data
    resolves, the Drell-Yan UFO model, and the external toolchains."""
    _doctor_report()
    if check_data:
        click.echo("")
        raise SystemExit(_check_data())


def _doctor_report() -> None:
    import os

    click.echo("\nPython modules:")
    for mod in ("numpy", "scipy", "matplotlib", "yaml", "click", "uproot",
                "mpi4py", "vegas", "pythia8"):
        ok = __import__("importlib.util", fromlist=["util"]).find_spec(mod) is not None
        click.secho(f"  {'OK ' if ok else 'NO '} {mod}", fg="green" if ok else "yellow")

    click.echo("\nData source repo (precomputed acceptance/brem/DY inputs):")
    env = os.environ.get("MCPSIM_SOURCE_REPO")
    how = (f"MCPSIM_SOURCE_REPO={env}" if env
           else f"default sibling location ({paths.source_repo()})")
    repo = paths.source_repo()
    if repo.exists():
        click.secho(f"  OK  resolved via {how}", fg="green")
    else:
        click.secho(f"  NO  resolved via {how} — does not exist. Download data.zip "
                    f"from https://doi.org/10.5281/zenodo.22690080, unzip it, and "
                    f"set MCPSIM_SOURCE_REPO to that folder.", fg="red")

    model = os.environ.get("MCPSIM_MCP_MODEL")
    click.echo("\nDrell-Yan UFO model (MCPSIM_MCP_MODEL):")
    if model and Path(model).exists():
        click.secho(f"  OK  {model}", fg="green")
    else:
        click.secho(f"  {'NO  set but missing: %s' % model if model else '-   unset'}"
                    "  — only needed for `mcpsim generate dy`.", fg="yellow")

    click.echo("\nExternal toolchains (not pip-installable):")
    _report_external_tool("root-config", "ROOT — needed to compile the C++ generators/decay.")
    _report_external_tool("pythia8-config", "PYTHIA 8 — needed for high-energy meson generation.")
    _report_external_tool("mpirun", "MPI runtime — needed for the VEGAS bremsstrahlung integrator.")
    _report_external_tool("mg5_aMC", "MadGraph5_aMC@NLO — needed for Drell-Yan generation.")


def _check_data() -> int:
    """Verify the resolved source repo against data/MANIFEST.yaml. Returns an
    exit code (0 = every listed file present with a matching sha256)."""
    import hashlib

    import yaml

    manifest_path = paths.PROJECT_ROOT / "data" / "MANIFEST.yaml"
    if not manifest_path.exists():
        click.secho(f"no manifest at {manifest_path} — your checkout is "
                    f"incomplete; re-clone or restore data/MANIFEST.yaml.",
                    fg="red")
        return 2
    try:
        repo = paths.require_source_repo()
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red")
        return 2

    entries = yaml.safe_load(manifest_path.read_text())["files"]
    missing = corrupt = ok = 0
    for e in entries:
        p = repo / e["path"]
        if not p.exists():
            click.secho(f"  MISSING {e['path']}  [{e['set']}]", fg="red")
            missing += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != e["sha256"]:
            click.secho(f"  CORRUPT {e['path']} (sha256 mismatch)", fg="red")
            corrupt += 1
        else:
            ok += 1
    color = "green" if not (missing or corrupt) else "red"
    click.secho(f"data check vs {repo}: {ok} ok, {missing} missing, "
                f"{corrupt} corrupt (of {len(entries)} listed)", fg=color)
    if missing and not corrupt:
        click.echo("  (a partial bundle is fine if the missing files belong "
                   "to sets you don't use — see the [set] tags above)")
    return 0 if not (missing or corrupt) else 1


def _report_external_tool(name: str, hint: str) -> None:
    import shutil

    found = shutil.which(name)
    if found:
        click.secho(f"  OK  {name}  ({found})", fg="green")
        return
    extra = _scan_conda_envs(name)
    if extra:
        click.secho(f"  -   {name}  not on PATH; found in conda env(s): {', '.join(extra)}",
                    fg="yellow")
        click.echo(f"        -> `conda activate {extra[0]}` then re-run mcpsim. {hint}")
    else:
        click.secho(f"  NO  {name}  — {hint}", fg="yellow")


def _scan_conda_envs(tool: str):
    """Return conda env names whose bin/ contains `tool`.

    Roots are found via $CONDA_EXE (any conda flavour: miniconda, anaconda,
    miniforge, mambaforge), falling back to the common home-directory names.
    """
    import os
    from pathlib import Path

    roots = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(Path(conda_exe).resolve().parent.parent)
    for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        root = Path.home() / name
        if root.exists() and root not in roots:
            roots.append(root)

    envs = []
    for root in roots:
        for env in [root] + sorted((root / "envs").glob("*")):
            if (env / "bin" / tool).exists():
                label = "base" if env == root else env.name
                if label not in envs:
                    envs.append(label)
    return envs


@cli.command("presets")
def presets_cmd() -> None:
    """List the built-in experiment presets."""
    names = list_presets()
    if not names:
        click.echo("(no presets found)")
        return
    click.echo(f"{'preset':12s}  {'beam':>9s}  {'frame':12s}  {'target':11s}  "
               f"{'detector':11s}  name")
    for name in names:
        cfg = load_preset(name)
        beam = f"{cfg.beam.energy_gev:g} GeV"
        click.echo(f"{name:12s}  {beam:>9s}  {cfg.beam.frame:12s}  "
                   f"{cfg.target.name:11s}  {cfg.detector.type:11s}  {cfg.name}")


def _execute(cfg: Config, *, out, channels, offaxis, plots: Sequence[str],
             regenerate: bool = False, mass_subset=None, gen_opts=None) -> None:
    """Shared body of run / production / limit: one pipeline run, or one per
    --offaxis angle plus overlay figures when there are several."""
    from .pipeline import run_pipeline
    if channels:
        cfg = replace(cfg, channels=_split(channels))
    try:
        angles = [float(x) for x in offaxis.split(",")] if offaxis else []
    except ValueError:
        raise click.BadParameter(f"expected comma-separated numbers, got {offaxis!r}",
                                 param_hint="--offaxis")

    runs = []
    for cfg_a in ([_with_offaxis(cfg, a) for a in angles] or [cfg]):
        sub = out and (f"{out}/{cfg_a.tag}" if len(angles) > 1 else out)
        with _clean_errors():
            result, written = run_pipeline(cfg_a, regenerate=regenerate, out_dir=sub,
                                           mass_subset=mass_subset, gen_opts=gen_opts,
                                           plots=plots)
        _report(cfg_a, written, result.warnings)
        runs.append((cfg_a.detector.offaxis_mrad, cfg_a, result))

    if len(runs) > 1:
        from .plotting.limits import plot_limit_overlay
        from .plotting.production import plot_production_overlay
        base_dir = Path(out).resolve() if out else paths.WORK_DIR / cfg.tag
        base_dir.mkdir(parents=True, exist_ok=True)
        overlays = {"production": plot_production_overlay, "limit": plot_limit_overlay}
        for name in plots:
            click.echo(f"  wrote {overlays[name](cfg, runs, base_dir)}")


@cli.command("run")
@_options(*_common, _channels_opt, *_engine_opts, _offaxis_opt)
@click.option("--regenerate", is_flag=True,
              help="Run full event generation (needs the ROOT/PYTHIA/MPI/MadGraph "
                   "toolchains; check with `mcpsim doctor`).")
@click.option("--mass-subset", type=int, default=None,
              help="With --regenerate: thin the mass grid to ~N points (smoke test).")
@click.option("--condor", is_flag=True,
              help="Regenerate brem and Drell-Yan as HTCondor batches (one job per mass) "
                   "instead of locally.")
@click.option("--no-submit", is_flag=True,
              help="With --condor, write the bundle but do not call condor_submit.")
def run_cmd(preset, config, out, channels, engine, fidelity, seed, quick, ngamma,
            target, offaxis, regenerate, mass_subset, condor, no_submit) -> None:
    """Compute and plot both the production and limit plots.

    Outputs (plots, numeric tables and a run_<tag>.yaml record of the resolved
    configuration) land in mcpsim-output/<experiment-tag>/ unless --out is
    given. With --offaxis a,b,c the run repeats per angle and overlay figures
    comparing the positions are added.
    """
    if mass_subset and not (regenerate or condor):
        raise click.UsageError("--mass-subset only applies with --regenerate/--condor "
                               "(fast mode always uses the full mass grid).")
    if no_submit and not condor:
        raise click.UsageError("--no-submit only applies with --condor.")
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma, target)
    _execute(cfg, out=out, channels=channels, offaxis=offaxis,
             plots=("production", "limit"), regenerate=regenerate or condor,
             mass_subset=mass_subset,
             gen_opts={"condor": True, "submit": not no_submit} if condor else None)


@cli.command("production")
@_options(*_common, _channels_opt, *_engine_opts, _offaxis_opt)
def production_cmd(preset, config, out, channels, engine, fidelity, seed, quick,
                   ngamma, target, offaxis) -> None:
    """Compute and plot only the production plot."""
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma, target)
    _execute(cfg, out=out, channels=channels, offaxis=offaxis, plots=("production",))


@cli.command("limit")
@_options(*_common, _channels_opt, *_engine_opts, _offaxis_opt)
def limit_cmd(preset, config, out, channels, engine, fidelity, seed, quick,
              ngamma, target, offaxis) -> None:
    """Compute and plot only the limit (exclusion) plot."""
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma, target)
    _execute(cfg, out=out, channels=channels, offaxis=offaxis, plots=("limit",))


@cli.command("compare")
@click.option("--preset", "-p", "presets", multiple=True,
              help="A preset to include; repeat it (-p darkquest -p ship).")
@click.option("--config", "-c", "configs", multiple=True,
              help="A YAML config to include; repeatable, mixes with --preset.")
@click.option("--out", "-o", "out", default=None,
              help="Output directory [default: mcpsim-output/compare/].")
@_options(_channels_opt, *_engine_opts)
def compare_cmd(presets, configs, out, channels, engine, fidelity, seed, quick,
                ngamma, target) -> None:
    """Overlay the limits of several runs on one figure.

    Each run is drawn solid with all channels and dashed without proton brem
    (the paper's DarkQuest + SHiP figure: -p darkquest -p ship). The tables,
    limit curve file and run record of every run are written next to it."""
    from .model import compute_production
    from .pipeline import write_tables
    from .plotting.limits import plot_limit_compare
    sources = [(p, None) for p in presets] + [(None, c) for c in configs]
    if not sources:
        raise click.UsageError("give at least one --preset or --config to compare.")
    out_dir = Path(out).resolve() if out else paths.WORK_DIR / "compare"
    runs = []
    for preset, config in sources:
        cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick,
                            ngamma, target)
        if channels:
            cfg = replace(cfg, channels=_split(channels))
        with _clean_errors():
            result = compute_production(cfg)
            written = write_tables(cfg, result, out_dir)
        _report(cfg, written, result.warnings)
        runs.append((cfg, result))
    name = "_vs_".join(c.tag for c, _ in runs)
    click.echo(f"  wrote {plot_limit_compare(runs, out_dir / f'limit_compare_{name}.pdf')}")


@cli.command("generate")
@click.argument("step", type=click.Choice(["mesons", "brem", "dy"]))
@_options(*_common, *_engine_opts)
@click.option("--mass-subset", type=int, default=None, help="Use only N masses.")
@click.option("--trials", type=int, default=None, help="Meson-generation trials.")
@click.option("--n-proc", type=int, default=None,
              help="Cores: MPI ranks for brem; per job under --condor (dy: MadGraph cores) "
                   "[default: the backend's own].")
@click.option("--brem-grid-size", type=int, default=None,
              help="VEGAS (theta,p) grid size [default: the backend's own].")
@click.option("--brem-neval", type=int, default=None,
              help="VEGAS evaluations per iteration [default: the backend's own].")
@click.option("--brem-nitn", type=int, default=None,
              help="VEGAS iterations [default: the backend's own].")
@click.option("--dy-nevents", type=int, default=None,
              help="MadGraph events per mass for dy (default 10000).")
@click.option("--condor", is_flag=True,
              help="brem/dy: emit/submit an HTCondor batch (one job per mass) instead of "
                   "running locally.")
@click.option("--no-submit", is_flag=True,
              help="With --condor, write the bundle but do not call condor_submit.")
def generate_cmd(step, preset, config, out, engine, fidelity, seed, quick,
                 ngamma, target, mass_subset, trials, n_proc, brem_grid_size, brem_neval,
                 brem_nitn, dy_nevents, condor, no_submit) -> None:
    """Run a single heavy generation step (mesons / brem / dy).

    Exits non-zero when the step produced no outputs (e.g. a missing
    toolchain); the reason is printed as a warning.
    """
    from .pipeline import regenerate_inputs
    if no_submit and not condor:
        raise click.UsageError("--no-submit only applies with --condor.")
    if condor and step == "mesons":
        raise click.UsageError("--condor applies to brem and dy.")
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma, target)
    # Restrict to the one requested channel so only that backend runs.
    channel = normalize_channels([step])[0]
    cfg = replace(cfg, channels=[channel])
    gen_dir = ((Path(out).resolve() if out else paths.WORK_DIR / cfg.tag) / "generated")
    gen_opts = {k: v for k, v in dict(
        trials=trials, n_proc=n_proc, brem_grid_size=brem_grid_size,
        brem_neval=brem_neval, brem_nitn=brem_nitn, dy_nevents=dy_nevents).items()
        if v is not None}
    if condor:
        gen_opts.update(condor=True, submit=not no_submit)
    with _clean_errors():
        new_cfg, gen_warnings = regenerate_inputs(cfg, gen_dir, mass_subset, gen_opts)
    for w in gen_warnings:
        click.secho(f"  ! {w}", fg="yellow")

    # Success = the requested channel's data reference now points into gen_dir.
    produced = {
        "meson_decay": lambda: any(str(gen_dir) in str(v)
                                   for v in new_cfg.data.acceptance.values()),
        "brem": lambda: bool(new_cfg.data.brem_dir
                             and str(gen_dir) in str(new_cfg.data.brem_dir)),
        "drell_yan": lambda: bool(new_cfg.data.dy_cross
                                  and str(gen_dir) in str(new_cfg.data.dy_cross)),
    }[channel]()
    if not produced:
        click.secho(f"Generation step '{step}' produced no outputs "
                    f"(see warnings above).", fg="red")
        raise SystemExit(1)

    if condor:
        click.echo(f"Generation step '{step}' is an HTCondor batch under {gen_dir}; "
                   f"its outputs appear once the jobs finish.")
    else:
        click.echo(f"Generation step '{step}' complete. Outputs under: {gen_dir}")
    # Ready-to-paste `data:` block so the outputs can be used by `mcpsim run`.
    d = new_cfg.data
    lines = {"brem": [f"  brem_dir: {d.brem_dir}", f"  brem_pattern: \"{d.brem_pattern}\""],
             "drell_yan": [f"  dy_cross: {d.dy_cross}", f"  dy_ageo: {d.dy_ageo}",
                           f"  dy_material: {d.dy_material}"],
             "meson_decay": ["  acceptance:"] + [f"    {k}: {v}" for k, v in d.acceptance.items()],
             }[channel]
    click.echo("Use them in a run by adding to your config:\ndata:\n" + "\n".join(lines))


def _split(channels: Optional[str]) -> Optional[List[str]]:
    if not channels:
        return None
    try:
        return normalize_channels(channels.split(","))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--channels")


def _report(cfg: Config, plots, warnings=None) -> None:
    click.echo(f"[{cfg.name}] {cfg.beam.energy_gev:g} GeV {cfg.beam.frame}, "
               f"{cfg.target.name} target, {cfg.detector.type}, channels={cfg.channels}")
    for w in (warnings or []):
        click.secho(f"  ! {w}", fg="yellow")
    for p in plots:
        click.echo(f"  wrote {p}")


if __name__ == "__main__":
    cli()
