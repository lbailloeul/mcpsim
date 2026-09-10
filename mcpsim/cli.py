"""mcpsim command-line interface.

    mcpsim presets                       list built-in experiment presets
    mcpsim run --preset darkquest        production + limit plots (fast mode)
    mcpsim run --config my.yaml          custom beam/target/detector
    mcpsim run --config my.yaml --regenerate   full PYTHIA/VEGAS/MadGraph
    mcpsim production / mcpsim limit      a single plot
    mcpsim generate mesons|brem|dy ...   one heavy generation step
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from . import paths
from .config import Config, list_presets, load_config, load_preset


def _resolve(preset: Optional[str], config: Optional[str]) -> Config:
    """Load the config and verify the data source repo before any compute."""
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


_common = [
    click.option("--preset", "-p", default=None, help="Built-in preset name (see `mcpsim presets`)."),
    click.option("--config", "-c", default=None, help="Path to a YAML config."),
    click.option("--out", "-o", "out", default=None,
                 help="Output directory [default: mcpsim-output/<experiment-tag>/]."),
]

_channels_opt = click.option("--channels", default=None,
                             help="Comma-separated subset of: meson_decay,brem,drell_yan.")

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
]


def _add_options(func):
    for opt in reversed(_common):
        func = opt(func)
    return func


def _add_engine_options(func):
    for opt in reversed(_engine_opts):
        func = opt(func)
    return func


def _apply_engine(cfg: Config, engine, fidelity, seed, quick, ngamma=None) -> Config:
    """Overlay CLI engine/sensitivity flags onto the config."""
    from dataclasses import replace
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
    return cfg


def _with_offaxis(cfg: Config, angle_mrad: float) -> Config:
    """Config displaced to one explicit off-axis angle.

    Always renames (so scan outputs land in per-angle directories, even for
    the angle matching the config default) and always recomputes acceptance —
    an explicit --offaxis request opts into the off-axis machinery."""
    from dataclasses import replace
    return replace(
        cfg,
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
    import os
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
        click.secho(f"  NO  resolved via {how} — does not exist. Set "
                    f"MCPSIM_SOURCE_REPO to your github-repo-scripts checkout "
                    f"or data bundle.", fg="red")

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


@cli.command("run")
@_add_options
@_channels_opt
@_add_engine_options
@click.option("--offaxis", default=None,
              help="Comma-separated off-axis angles in mrad (e.g. '4,6.5,9'); "
                   "one run per angle plus overlay plots.")
@click.option("--regenerate", is_flag=True,
              help="Run full event generation (needs the ROOT/PYTHIA/MPI/MadGraph "
                   "toolchains; check with `mcpsim install --dry-run`).")
@click.option("--mass-subset", type=int, default=None,
              help="With --regenerate: thin the mass grid to ~N points (smoke test).")
@click.option("--condor", is_flag=True,
              help="Regenerate brem via an HTCondor batch (one job/mass) instead of local mpirun.")
@click.option("--no-submit", is_flag=True,
              help="With --condor, write the bundle but do not call condor_submit.")
def run_cmd(preset, config, out, channels, engine, fidelity, seed, quick, ngamma,
            offaxis, regenerate, mass_subset, condor, no_submit) -> None:
    """Compute and plot both the production and limit plots.

    Outputs (plots + numeric tables) land in mcpsim-output/<experiment-tag>/
    unless --out is given. With --offaxis a,b,c the run repeats per angle and
    overlay figures comparing the positions are added.
    """
    from .pipeline import run_pipeline
    if mass_subset and not (regenerate or condor):
        raise click.UsageError("--mass-subset only applies with --regenerate/--condor "
                               "(fast mode always uses the full mass grid).")
    if no_submit and not condor:
        raise click.UsageError("--no-submit only applies with --condor.")
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma)
    gen_opts = {"condor": True, "submit": not no_submit} if condor else None

    angles = [float(x) for x in offaxis.split(",")] if offaxis else None
    if not angles:
        with _clean_errors():
            result, plots = run_pipeline(
            cfg, regenerate=regenerate or condor,
            channels=_split(channels), out_dir=out, mass_subset=mass_subset,
            gen_opts=gen_opts,
        )
        _report(cfg, plots, result.warnings)
        return

    runs = []
    for a in angles:
        cfg_a = _with_offaxis(cfg, a)
        with _clean_errors():
            result, plots = run_pipeline(
            cfg_a, regenerate=regenerate or condor,
            channels=_split(channels), out_dir=out and f"{out}/{cfg_a.tag}",
            mass_subset=mass_subset, gen_opts=gen_opts,
        )
        _report(cfg_a, plots, result.warnings)
        runs.append((a, cfg_a, result))

    if len(runs) > 1:
        from .plotting.limits import plot_limit_overlay
        from .plotting.production import plot_production_overlay
        base_dir = Path(out).resolve() if out else paths.WORK_DIR / cfg.tag
        base_dir.mkdir(parents=True, exist_ok=True)
        for fn in (plot_production_overlay, plot_limit_overlay):
            p = fn(cfg, runs, base_dir)
            click.echo(f"  wrote {p}")


@cli.command("production")
@_add_options
@_channels_opt
@_add_engine_options
@click.option("--offaxis", type=float, default=None,
              help="Single off-axis angle in mrad (use `run --offaxis a,b,c` for scans).")
def production_cmd(preset, config, out, channels, engine, fidelity, seed, quick,
                   ngamma, offaxis) -> None:
    """Compute and plot only the production plot."""
    from .model import compute_production
    from .plotting.production import plot_production
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma)
    if offaxis is not None:
        cfg = _with_offaxis(cfg, offaxis)
    if channels:
        from dataclasses import replace
        cfg = replace(cfg, channels=_split(channels))
    with _clean_errors():
        result = compute_production(cfg)
    out_dir = Path(out).resolve() if out else paths.WORK_DIR / cfg.tag
    from .pipeline import write_tables
    tables = write_tables(cfg, result, out_dir)
    path = plot_production(cfg, result, out_dir / f"production_{cfg.tag}.pdf")
    _report(cfg, [path, *tables], result.warnings)


@cli.command("limit")
@_add_options
@_channels_opt
@_add_engine_options
@click.option("--offaxis", type=float, default=None,
              help="Single off-axis angle in mrad (use `run --offaxis a,b,c` for scans).")
def limit_cmd(preset, config, out, channels, engine, fidelity, seed, quick,
              ngamma, offaxis) -> None:
    """Compute and plot only the limit (exclusion) plot."""
    from .model import compute_production
    from .plotting.limits import plot_limit
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma)
    if offaxis is not None:
        cfg = _with_offaxis(cfg, offaxis)
    if channels:
        from dataclasses import replace
        cfg = replace(cfg, channels=_split(channels))
    with _clean_errors():
        result = compute_production(cfg)
    out_dir = Path(out).resolve() if out else paths.WORK_DIR / cfg.tag
    from .pipeline import write_tables
    tables = write_tables(cfg, result, out_dir)
    path = plot_limit(cfg, result, out_dir / f"limit_{cfg.tag}.pdf")
    _report(cfg, [path, *tables], result.warnings)


@cli.command("generate")
@click.argument("step", type=click.Choice(["mesons", "brem", "dy"]))
@_add_options
@_add_engine_options
@click.option("--mass-subset", type=int, default=None, help="Use only N masses.")
@click.option("--trials", type=int, default=None, help="Meson-generation trials.")
@click.option("--n-proc", type=int, default=4, help="MPI ranks for brem (per job under --condor).")
@click.option("--brem-grid-size", type=int, default=100, help="VEGAS (theta,p) grid size.")
@click.option("--brem-neval", type=int, default=4000, help="VEGAS evaluations per iteration.")
@click.option("--brem-nitn", type=int, default=10, help="VEGAS iterations.")
@click.option("--condor", is_flag=True,
              help="brem: emit/submit an HTCondor batch (one job/mass) instead of local mpirun.")
@click.option("--no-submit", is_flag=True,
              help="With --condor, write the bundle but do not call condor_submit.")
def generate_cmd(step, preset, config, out, engine, fidelity, seed, quick,
                 ngamma, mass_subset, trials, n_proc, brem_grid_size, brem_neval,
                 brem_nitn, condor, no_submit) -> None:
    """Run a single heavy generation step (mesons / brem / dy).

    Exits non-zero when the step produced no outputs (e.g. a missing
    toolchain); the reason is printed as a warning.
    """
    from .pipeline import regenerate_inputs
    if no_submit and not condor:
        raise click.UsageError("--no-submit only applies with --condor.")
    cfg = _apply_engine(_resolve(preset, config), engine, fidelity, seed, quick, ngamma)
    # Restrict to the one requested channel so only that backend runs.
    from dataclasses import replace
    channel = {"mesons": "meson_decay", "brem": "brem", "dy": "drell_yan"}[step]
    cfg = replace(cfg, channels=[channel])
    gen_dir = ((Path(out).resolve() if out else paths.WORK_DIR / cfg.tag) / "generated")
    gen_opts = {k: v for k, v in dict(
        trials=trials, n_proc=n_proc, brem_grid_size=brem_grid_size,
        brem_neval=brem_neval, brem_nitn=brem_nitn).items() if v is not None}
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

    click.echo(f"Generation step '{step}' complete. Outputs under: {gen_dir}")
    if channel == "brem" and new_cfg.data.brem_dir:
        click.echo(f"  brem_dir = {new_cfg.data.brem_dir}")
    if channel == "drell_yan" and new_cfg.data.dy_cross:
        click.echo(f"  dy_cross = {new_cfg.data.dy_cross}")
    if channel == "meson_decay":
        for k, v in new_cfg.data.acceptance.items():
            click.echo(f"  acceptance[{k}] = {v}")


def _split(channels: Optional[str]):
    return [c.strip() for c in channels.split(",")] if channels else None


def _report(cfg: Config, plots, warnings=None) -> None:
    click.echo(f"[{cfg.name}] {cfg.beam.energy_gev:g} GeV {cfg.beam.frame}, "
               f"{cfg.detector.type}, channels={cfg.channels}")
    for w in (warnings or []):
        click.secho(f"  ! {w}", fg="yellow")
    for p in plots:
        click.echo(f"  wrote {p}")


if __name__ == "__main__":
    cli()
