"""Production plot: N_chi/eps^2 vs mass for every channel.

Modeled on DrawProductionDarkQuest.py (the matplotlib styling, log-log axes and
per-channel curves) but driven by a ProductionResult and Config.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import Config  # noqa: E402
from ..model import ProductionResult  # noqa: E402
from ..physics.constants import MESONS  # noqa: E402

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


def plot_production(cfg: Config, result: ProductionResult, output: str | Path) -> Path:
    """Render and save the production plot. Returns the output path."""
    m = result.masses
    fig, ax = plt.subplots(figsize=(8.09, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\chi}\, [\mathrm{GeV}/c^{2}]$", fontsize=18)
    ax.set_ylabel(r"$N_{\chi}/\epsilon^{2}$", fontsize=18)

    lw = 3.0
    for name, spec in MESONS.items():
        y = result.meson_channels.get(name)
        if y is None:
            continue
        ax.plot(m, _mask(y), lw=lw, label=spec.label)

    if result.drell_yan is not None:
        ax.plot(m, _mask(result.drell_yan), lw=lw,
                label=r"$q\overline{q}\to\gamma^{*}\to\chi\overline{\chi}$")

    if result.brem_central is not None and result.brem_central.sum() > 0:
        ax.plot(m, _mask(result.brem_central), color="#bcbd22", lw=lw, ls="--",
                label=r"$p$ brem ($\Lambda_p = 1.5~\mathrm{GeV}$)", zorder=2)
        if result.brem_low is not None and result.brem_high is not None:
            ax.fill_between(m, _mask(result.brem_low), _mask(result.brem_high),
                            color="#bcbd22", alpha=0.5, linewidth=0.0, zorder=1)

    ax.plot(m, _mask(result.total), color="black", lw=lw, label="Total")

    _set_decade_limits(ax, result.total)
    ax.legend(loc="lower left", ncol=2, frameon=True, fancybox=True,
              framealpha=0.85, fontsize=11, handlelength=2.2)

    _annotate(ax, cfg)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output


def plot_production_overlay(cfg: Config, runs, out_dir: Path) -> Path:
    """Overlay of total production (N_chi/eps^2) for an off-axis scan.

    `runs` is a list of (angle_mrad, cfg_at_angle, ProductionResult)."""
    from .limits import OFFAXIS_RAMP

    fig, ax = plt.subplots(figsize=(8.09, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    colors = (OFFAXIS_RAMP if len(runs) <= len(OFFAXIS_RAMP)
              else [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, len(runs))])
    for color, (angle, _cfg_a, result) in zip(colors, runs):
        y = _mask(result.total)
        ax.plot(result.masses, y, color=color, lw=2, label=f"{angle:g} mrad")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=14)
    ax.set_ylabel(r"$N_\chi/\epsilon^2$", fontsize=14)
    ax.text(0.03, 0.03, f"{cfg.beam.n_pot:.2g} POT @ {cfg.beam.energy_gev:g} GeV, "
            f"{cfg.detector.distance_m:g} m", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=10, color="0.35")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=11)

    out = Path(out_dir) / f"production_overlay_{cfg.tag}.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _mask(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y[y <= 0] = np.nan
    return y


def _set_decade_limits(ax, total: np.ndarray) -> None:
    pos = total[total > 0]
    if pos.size == 0:
        return
    top = 10 ** np.ceil(np.log10(pos.max()))
    ax.set_ylim(max(top / 1e14, 1.0), top)


def _annotate(ax, cfg: Config) -> None:
    det = cfg.detector
    if det.type == "cylindrical":
        geo = f"{det.distance_m:g} m, {det.radius_m:g} m radius cylindrical detector"
    else:
        geo = (f"{det.distance_m:g} m, {det.bar_columns}x{det.bar_rows}x{det.bar_layers} "
               f"bar array")
    txt = (f"{cfg.name}: {cfg.beam.n_pot:.0e} POT, {cfg.beam.energy_gev:g} GeV\n{geo}")
    ax.text(0.5, 0.93, txt, transform=ax.transAxes, ha="left", va="center",
            fontsize=11.5,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8))
