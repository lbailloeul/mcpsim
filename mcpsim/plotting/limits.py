"""Limit (exclusion) plots: the eps(m) at which the signal reaches the
threshold, over the existing constraints.

Styled after the paper's final figure (plotting-scripts-backup/
DrawProbability_final_*.py), with the existing constraints kept as one merged
grey region: every run is a solid curve (all channels) over a dashed lighter
one without proton brem. plot_limit draws one run, plot_limit_compare several
(the paper's DarkQuest + SHiP figure), plot_limit_overlay an off-axis scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths  # noqa: E402
from ..config import Config  # noqa: E402
from ..model import ProductionResult  # noqa: E402
from ..physics import sensitivity  # noqa: E402
from .style import family_colors, log_decades, paper_style, pot_tex, save  # noqa: E402

# (filename, mass-unit-is-MeV) for the existing-constraint overlays.
CONSTRAINTS: List[Tuple[str, bool]] = [
    ("MiniBooNE.csv", False),
    ("SLACmQ.csv", False),
    ("BEBC.csv", False),
    ("LSND.csv", False),
    ("CHARMII.csv", False),
    ("ArgoNeuT.csv", False),
    ("SENSEI_POLISHED.csv", True),
    ("milliQanRun3Fix.csv", False),
]
# Sequential single-hue ramp for ordered off-axis positions (light -> dark).
OFFAXIS_RAMP = ["#B3CDE8", "#7FAFDA", "#4E8FC7", "#2A6BAB", "#0F4C81"]
X_MAX_GEV = 11.2                 # the paper's upper mass edge
Y_TOP = 2.0


def limit_curves(result: ProductionResult):
    """(eps with all channels, eps without proton brem or None) on the mass grid."""
    s, brem = result.config.sensitivity, result.brem_central
    no_brem = (sensitivity.exclusion_contour(result.total - brem, s)
               if brem is not None and brem.sum() > 0 else None)
    return sensitivity.exclusion_contour(result.total, s), no_brem


def plot_limit(result: ProductionResult, output: str | Path) -> Path:
    """One run over the existing constraints (PDF + PNG). Returns the PDF path."""
    return plot_limit_compare([result], output)


def plot_limit_compare(results: Sequence[ProductionResult], output: str | Path) -> Path:
    """Several runs on one exclusion plot, each solid with all channels and
    dashed without proton brem (the paper's DarkQuest + SHiP figure)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(11, 8))
        _draw_constraints(ax)
        used: set = set()
        reach = []
        for result in results:
            cfg = result.config
            solid, light = family_colors(cfg.family, used)
            eps, eps_no_brem = limit_curves(result)
            label = f"{cfg.name}: {pot_tex(cfg.beam.n_pot)} POT"
            if eps_no_brem is not None:
                _curve(ax, result.masses, eps_no_brem, light, 1.8, (0, (4, 2.5)),
                       f"{label}, without $p$ brem")
                label += ", including $p$ brem"
            ok = _curve(ax, result.masses, eps, solid, 2.4, "-", label)
            if ok is not None:
                reach.append(ok)
        _frame(ax, results, reach)
        ax.legend(loc="upper left", fontsize=14, framealpha=1, edgecolor="0.8")
        fig.tight_layout()
        return save(fig, output)


def plot_limit_overlay(cfg: Config, results: Sequence[ProductionResult],
                       out_dir: Path) -> Path:
    """eps(m) for an off-axis scan of `cfg`, one result per angle, light to
    dark with angle, over the constraints, with the view following the
    curves."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(9, 6.5))
        env_m, env_y = _draw_constraints(ax)
        colors = (OFFAXIS_RAMP if len(results) <= len(OFFAXIS_RAMP)
                  else [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, len(results))])
        curves = []
        for color, result in zip(colors, results):
            eps, _ = limit_curves(result)
            angle = result.config.detector.offaxis_mrad
            if _curve(ax, result.masses, eps, color, 2.0, "-", f"{angle:g} mrad off axis") is not None:
                curves.append((result.masses[np.isfinite(eps)], eps[np.isfinite(eps)]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        if curves:
            xlo = min(c[0].min() for c in curves) * 0.9
            xhi = max(c[0].max() for c in curves) * 1.25
            ylo = min(c[1].min() for c in curves)
            if env_y is not None:
                win = (env_m >= xlo) & (env_m <= xhi)
                if np.any(win):
                    ylo = min(ylo, float(env_y[win].min()))
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo * 0.8, max(c[1].max() for c in curves) * 2.2)
        log_decades(ax, "x")
        log_decades(ax, "y")
        ax.set_xlabel(r"$m_{\chi}$ [$\mathrm{GeV}/\mathrm{c}^2$]", fontsize=18)
        ax.set_ylabel(r"$\epsilon=Q/e$", fontsize=18)
        ax.legend(frameon=False, loc="upper left", fontsize=12.5)
        _cl_note(ax, cfg, extra=f"{pot_tex(cfg.beam.n_pot)} POT at "
                                f"${cfg.beam.energy_gev:g}$ GeV, ${cfg.detector.distance_m:g}$ m")
        fig.tight_layout()
        out = Path(out_dir) / f"limit_overlay_{cfg.tag}.pdf"
        return save(fig, out)


def _curve(ax, m, eps, color, lw, ls, label) -> Optional[float]:
    """Draw one eps(m) boundary; returns its last reached mass (None if none)."""
    ok = np.isfinite(eps)
    if not np.any(ok):
        # the signal never reaches the threshold: say so instead of drawing nothing
        ax.plot([], [], color=color, lw=lw, ls=ls, label=f"{label} (never reaches threshold)")
        return None
    ax.plot(m[ok], eps[ok], color=color, lw=lw, ls=ls, label=label, zorder=5)
    return float(m[ok].max())


def _frame(ax, results: Sequence[ProductionResult], reach: List[float]) -> None:
    cfg0 = results[0].config
    s = cfg0.sensitivity
    m_min = min(float(r.masses.min()) for r in results)
    m_max = max(float(r.masses.max()) for r in results)
    # the paper's 11.2 GeV edge, or close in on a curve that ends early
    xhi = min(X_MAX_GEV, m_max * 1.6, 1.5 * max(reach)) if reach else m_max
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(1e-3, m_min), xhi)
    ax.set_ylim(min(r.config.sensitivity.charge_min for r in results), s.charge_max)
    log_decades(ax, "x")
    log_decades(ax, "y")
    ax.set_xlabel(r"$m_{\chi}$ [$\mathrm{GeV}/\mathrm{c}^2$]", fontsize=22)
    ax.set_ylabel(r"$\epsilon=Q/e$", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    _cl_note(ax, cfg0)


def _cl_note(ax, cfg: Config, extra: str = "") -> None:
    thr = cfg.sensitivity.n_chi_threshold
    note = (rf"95% CL, zero background ($N_\chi = {thr:g}$)" if abs(thr - 3.09) < 1e-9
            else rf"excluded where $N_\chi \geq {thr:g}$ events")
    if extra:
        note = f"{extra}\n{note}"
    ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, color="0.25")


def _draw_constraints(ax):
    """Shade the union of existing constraints as one grey region. Returns
    (m_grid, envelope), envelope None when no contour file was found, so
    callers can fold it into their axis ranges."""
    m_grid = np.logspace(np.log10(0.0009916136673313), np.log10(11.3), 800)
    env = np.full_like(m_grid, Y_TOP)
    found = False
    contours_dir = paths.contours_dir()
    for fname, is_mev in CONSTRAINTS:
        arr = _load_csv(contours_dir / fname)
        if arr is None:
            continue
        x, y = arr[:, 0], arr[:, 1]
        if is_mev:
            x = x / 1000.0
        valid = (x > 0) & (y > 0)
        if not np.any(valid):
            continue
        found = True
        order = np.argsort(x[valid])
        xl, yl = np.log10(x[valid][order]), np.log10(y[valid][order])
        yq = 10 ** np.interp(np.log10(m_grid), xl, yl,
                             left=np.log10(Y_TOP), right=np.log10(Y_TOP))
        env = np.minimum(env, yq)
    if not found:
        return m_grid, None
    ax.fill_between(m_grid, env, Y_TOP, color="lightgray", alpha=0.7,
                    label="Existing constraints", zorder=1)
    return m_grid, env


def _load_csv(path: Path):
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr
