"""Limit (exclusion) plot: N_chi = threshold contour in the (m_chi, epsilon) plane.

The signal is the baseline yield broadcast with the light-yield detection factor
(sensitivity.signal_2d). Existing experimental constraints from
experiment-contours-small/*.csv are shaded as an envelope, following
DrawProbability.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths  # noqa: E402
from ..config import Config  # noqa: E402
from ..model import ProductionResult  # noqa: E402
from ..physics import sensitivity  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

# (filename, label, mass-unit-is-MeV) for the existing-constraint overlays.
CONSTRAINTS: List[Tuple[str, str, bool]] = [
    ("MiniBooNE.csv", "MiniBooNE", False),
    ("SLACmQ.csv", "SLAC mQ", False),
    ("BEBC.csv", "BEBC", False),
    ("LSND.csv", "LSND", False),
    ("CHARMII.csv", "CHARM II", False),
    ("ArgoNeuT.csv", "ArgoNeuT", False),
    ("SENSEI_POLISHED.csv", "SENSEI", True),
    ("milliQanRun3Fix.csv", "milliQan", False),
]

_FAMILY_COLOR = {"darkquest": "#C94F65", "ship": "#0487FF", "lanl": "tomato"}


def plot_limit(cfg: Config, result: ProductionResult, output: str | Path) -> Path:
    """Exclusion plot: the eps95(m) boundary over the existing-constraint envelope."""
    s = cfg.sensitivity
    charges = sensitivity.charge_grid(s.charge_min, s.charge_max, s.n_charge)
    eps95 = sensitivity.exclusion_contour(result.total, charges, s.n_gamma, s.a,
                                          s.n_chi_threshold)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    _draw_constraint_envelope(ax)

    color = _FAMILY_COLOR.get(cfg.family, "#C94F65")
    label = f"{cfg.name}: {cfg.beam.n_pot:.0e} POT"
    reached = np.isfinite(eps95)
    if np.any(reached):
        ax.plot(result.masses[reached], eps95[reached], color=color, lw=3, label=label)
    else:
        # Previously ax.contour silently drew nothing here; make it explicit.
        ax.text(0.5, 0.5, "signal never reaches the exclusion threshold\n"
                f"(N_chi = {s.n_chi_threshold:g}) anywhere on the grid",
                transform=ax.transAxes, ha="center", va="center", fontsize=16,
                color=color)
        ax.plot([], [], color=color, lw=3, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\chi}$ [$\mathrm{GeV}/\mathrm{c}^2$]", fontsize=22)
    ax.set_ylabel(r"$\epsilon=Q/e$", fontsize=22)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(max(1e-3, result.masses.min()), result.masses.max())
    ax.set_ylim(s.charge_min, s.charge_max)
    ax.legend(loc="upper left", fontsize=16)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# Sequential single-hue ramp for ordered off-axis positions (light -> dark).
OFFAXIS_RAMP = ["#B3CDE8", "#7FAFDA", "#4E8FC7", "#2A6BAB", "#0F4C81"]


def plot_limit_overlay(cfg: Config, runs, out_dir: Path) -> Path:
    """Overlay of eps95(m) exclusion boundaries for an off-axis scan.

    `runs` is a list of (angle_mrad, cfg_at_angle, ProductionResult); curves
    are drawn light-to-dark with increasing angle over the existing-constraint
    envelope, with view ranges following the curves (the envelope would
    otherwise autoscale to eps = 2)."""
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=200)
    env_m, env_y = _draw_constraint_envelope(ax)

    s = cfg.sensitivity
    charges = sensitivity.charge_grid(s.charge_min, s.charge_max, s.n_charge)
    curves = []
    colors = (OFFAXIS_RAMP if len(runs) <= len(OFFAXIS_RAMP)
              else [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, len(runs))])
    for color, (angle, cfg_a, result) in zip(colors, runs):
        eps95 = sensitivity.exclusion_contour(result.total, charges, s.n_gamma,
                                              s.a, s.n_chi_threshold)
        m = np.isfinite(eps95)
        if np.any(m):
            ax.plot(result.masses[m], eps95[m], color=color, lw=2,
                    label=f"{angle:g} mrad")
            curves.append((result.masses[m], eps95[m]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\chi}$ [GeV]", fontsize=16)
    ax.set_ylabel(r"$\epsilon=Q/e$", fontsize=16)
    ax.text(0.97, 0.03, f"{cfg.beam.n_pot:.2g} POT @ {cfg.beam.energy_gev:g} GeV, "
            f"{cfg.detector.distance_m:g} m", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=10, color="0.35")
    if curves:
        xlo = min(c[0].min() for c in curves) * 0.9
        xhi = max(c[0].max() for c in curves) * 1.25
        ylo = min(c[1].min() for c in curves)
        if env_y is not None:
            win = (env_m >= xlo) & (env_m <= xhi)
            if np.any(win):
                ylo = min(ylo, float(env_y[win].min()))
        yhi = max(c[1].max() for c in curves) * 2.2
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo * 0.8, yhi)
    ax.legend(frameon=False, loc="upper left", fontsize=11)

    out = Path(out_dir) / f"limit_overlay_{cfg.tag}.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_constraint_envelope(ax, y_top: float = 2.0):
    """Shade the union of existing constraints as a single gray envelope.

    Returns (m_grid, envelope) — envelope is None when no contour was found —
    so callers can fold the envelope into their axis ranges."""
    m_grid = np.logspace(np.log10(1e-3), np.log10(11.3), 800)
    env = np.full_like(m_grid, y_top)
    found = False
    contours_dir = paths.contours_dir()
    for fname, _label, is_mev in CONSTRAINTS:
        path = contours_dir / fname
        arr = _load_csv(path)
        if arr is None:
            continue
        found = True
        x, y = arr[:, 0], arr[:, 1]
        if is_mev:
            x = x / 1000.0
        valid = (x > 0) & (y > 0)
        if not np.any(valid):
            continue
        order = np.argsort(x[valid])
        xl, yl = np.log10(x[valid][order]), np.log10(y[valid][order])
        yq = 10 ** np.interp(np.log10(m_grid), xl, yl,
                             left=np.log10(y_top), right=np.log10(y_top))
        env = np.minimum(env, yq)
    if found:
        ax.fill_between(m_grid, env, y_top, color="lightgray", alpha=0.7,
                        label="Existing constraints")
    return m_grid, (env if found else None)


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
