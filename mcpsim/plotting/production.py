"""Production plot: N_chi/eps^2 vs mass for every channel.

The paper's final figure (plotting-scripts-backup/DrawProduction{DarkQuest,
SHIP}_final.py) driven by a ProductionResult: the same channel colours, the
grey Lambda_p = 1-2 GeV brem band with the 1.5 GeV line, the brem "matched"
to the other channels (capped at their sum) in red dashes, a dotted total,
Drell-Yan drawn from m = M_min/2 = 1 GeV (below it the channel only counts
pairs above the 2 GeV mass cut; the total keeps it), and curves that drop to
the axis at their kinematic endpoints. The legend, headed by the run
description, sits beside the axes so it never covers a curve (in the paper
it hides the Upsilon line).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import geometry  # noqa: E402
from ..config import Config, run_config  # noqa: E402
from ..model import ProductionResult  # noqa: E402
from ..physics.constants import MESONS  # noqa: E402
from .style import log_decades, paper_style, pot_tex, save  # noqa: E402

CHANNEL_COLORS = {"pi0": "#1f77b4", "eta": "#ff7f0e", "jpsi": "#2ca02c", "upsilon": "#a4a41e",
                  "rho": "#9467bd", "omega": "#8c564b", "phi": "#e377c2"}
DY_COLOR, BREM_GRAY = "#d62728", "#7f7f7f"
DY_LABEL = r"$q\overline{q}\to\gamma^{*}\to\chi\overline{\chi}$"
DY_PLOT_FROM_GEV = 1.0          # half the 2 GeV pair-mass cut of the DY inputs


def plot_production(cfg: Config, result: ProductionResult, output: str | Path) -> Path:
    """Render and save the production plot (PDF + PNG). Returns the PDF path."""
    cfg = run_config(cfg)
    m = result.masses
    with paper_style():
        fig, ax = plt.subplots(figsize=(8.09, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.margins(x=0, y=0)
        ax.set_xlabel(r"$m_{\chi}\, [\mathrm{GeV}/c^{2}]$", fontsize=18)
        ax.set_ylabel(r"$N_{\chi}/\epsilon^{2}$", fontsize=18)

        lw = 1.8
        for name, color in CHANNEL_COLORS.items():
            y = result.meson_channels.get(name)
            if y is not None:
                ax.plot(m, _endpoint(y), lw=lw, color=color, label=MESONS[name].label)
        if result.drell_yan is not None:
            ax.plot(m, _endpoint(np.where(m >= DY_PLOT_FROM_GEV, result.drell_yan, np.nan)),
                    lw=lw, color=DY_COLOR, label=DY_LABEL)

        brem = result.brem_central
        has_brem = brem is not None and brem.sum() > 0
        if has_brem:
            ax.plot(m, _endpoint(brem), color=BREM_GRAY, lw=1.8,
                    label=r"$p$ brem ($\Lambda_p = 1.5~\mathrm{GeV}$)", zorder=4)
            if result.brem_low is not None and result.brem_high is not None:
                ax.fill_between(m, _positive(result.brem_low), _positive(result.brem_high),
                                color=BREM_GRAY, alpha=0.5, linewidth=0.0, zorder=1,
                                label=r"$p$ brem, $\Lambda_p \in [1,2]~\mathrm{GeV}$")
            # matched: brem capped at the other channels' sum (paper)
            matched = np.minimum(brem, result.total - brem)
            ax.plot(m, _endpoint(matched), color="red", lw=1.8, ls=(4, (4, 4)),
                    label=r"matched $p$ brem", zorder=4)
        ax.plot(m, _endpoint(result.total), color="black", lw=2.5, ls=(0, (1, 1.5)),
                label="Total", zorder=5)

        _limits(ax, m, result.total)
        _legend(ax, _description(cfg))
        # no tight_layout: the axes keep the paper's proportions and the saved
        # figure grows (bbox_inches="tight") to take in the legend column
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.97)
        return save(fig, output)


def plot_production_overlay(cfg: Config, runs, out_dir: Path) -> Path:
    """Total production for an off-axis scan, light to dark with angle.

    `runs` is a list of (angle_mrad, cfg_at_angle, ProductionResult)."""
    from .limits import OFFAXIS_RAMP

    with paper_style():
        fig, ax = plt.subplots(figsize=(8.09, 5.5))
        colors = (OFFAXIS_RAMP if len(runs) <= len(OFFAXIS_RAMP)
                  else [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, len(runs))])
        for color, (angle, _cfg_a, result) in zip(colors, runs):
            ax.plot(result.masses, _positive(result.total), color=color, lw=2,
                    label=f"{angle:g} mrad")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\chi}\, [\mathrm{GeV}/c^{2}]$", fontsize=18)
        ax.set_ylabel(r"$N_{\chi}/\epsilon^{2}$", fontsize=18)
        log_decades(ax, "x")
        log_decades(ax, "y")
        ax.legend(frameon=False, fontsize=12.5, title="off-axis angle", title_fontsize=12.5)
        ax.text(0.03, 0.04, _exposure_line(cfg) + f", ${cfg.detector.distance_m:g}$ m",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5)
        fig.tight_layout()
        out = Path(out_dir) / f"production_overlay_{cfg.tag}.pdf"
        return save(fig, out)


def _endpoint(y: np.ndarray) -> np.ndarray:
    """Blank a channel before it starts, but keep its zeros after the end:
    on the log axis those draw the drop to the axis at the kinematic
    endpoint, as in the paper."""
    y = np.asarray(y, dtype=float).copy()
    pos = np.flatnonzero(y > 0)
    if pos.size == 0:
        return np.full_like(y, np.nan)
    y[:pos[0]] = np.nan
    return y


def _positive(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y[y <= 0] = np.nan
    return y


def _limits(ax, m: np.ndarray, total: np.ndarray) -> None:
    """y over 14 decades below one decade of headroom (room for the label);
    x from the first mass to the decade above where the total leaves the
    plot."""
    pos = total > 0
    if not np.any(pos):
        return
    top = 10.0 ** (np.ceil(np.log10(total[pos].max())) + 1)
    bottom = top / 1e14
    ax.set_ylim(bottom, top)
    shown = total >= bottom
    ax.set_xlim(m.min(), 10.0 ** np.ceil(np.log10(m[shown].max() if np.any(shown) else m.max())))
    log_decades(ax, "x", skip_low=True)
    log_decades(ax, "y", every=2)


def _legend(ax, title: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    # band entry right under the 1.5 GeV brem line, as in the paper
    if "matched $p$ brem" in labels and any("in [1,2]" in lab for lab in labels):
        band = next(i for i, lab in enumerate(labels) if "in [1,2]" in lab)
        line = labels.index(r"$p$ brem ($\Lambda_p = 1.5~\mathrm{GeV}$)")
        handles.insert(line + 1, handles.pop(band))
        labels.insert(line + 1, labels.pop(band))
    # the run description heads the legend: beside the axes, it never covers a curve
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=False, fontsize=11.5, handlelength=2.4, labelspacing=0.35,
                    borderaxespad=0.0, borderpad=0.0, title=title, title_fontsize=11.5,
                    alignment="left")
    leg.get_title().set_linespacing(1.3)
    # the matched line's dash phase offset (so its dashes interleave with the
    # grey brem line) would start its legend swatch mid-gap: reset it there
    line_labels = [lab for h, lab in zip(handles, labels) if isinstance(h, plt.Line2D)]
    for hline, lab in zip(leg.get_lines(), line_labels):
        if lab == "matched $p$ brem":
            hline.set_linestyle((0, (4, 4)))


def _exposure_line(cfg: Config) -> str:
    return f"{pot_tex(cfg.beam.n_pot)} POT, ${cfg.beam.energy_gev:g}$ GeV"


def _geometry_line(cfg: Config) -> str:
    det = cfg.detector
    if det.type == "cylindrical":
        geo = f"${det.distance_m:g}$ m, ${det.radius_m:g}$ m radius cylindrical detector"
    else:
        width, height = geometry.bar_array_dims(det)
        geo = f"${det.distance_m:g}$ m, ${width:g}\\times{height:g}$ m face"
    if det.offaxis_mrad:
        geo += f", ${det.offaxis_mrad:g}$ mrad off axis"
    return geo


def _description(cfg: Config) -> str:
    name = cfg.name
    if cfg.engine.fidelity == "legacy" and "legacy" not in name.lower():
        name += " (legacy)"
    return f"{name}\n{_exposure_line(cfg)} $p$ on {cfg.target.name}\n{_geometry_line(cfg)}"
