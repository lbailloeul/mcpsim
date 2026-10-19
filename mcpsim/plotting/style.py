"""The look of the paper's final figures, shared by every mcpsim plot.

Taken from plotting-scripts-backup/DrawProductionDarkQuest_final.py,
DrawProductionSHIP_final.py and DrawProbability_final_*.py: Computer Modern
mathtext on a serif face, inward ticks on all four sides with minor ticks,
decade labels written 10^n (and 1 for 10^0), exposures typeset as 10^20.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

PAPER_RC = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 7, "xtick.minor.size": 3.5,
    "ytick.major.size": 7, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
}

# The paper's experiment colours (solid: with brem, light: without).
FAMILY_COLORS = {
    "darkquest": ("#FF7F0E", "#FFAC62"),
    "ship": ("#0072B2", "#66AAD0"),
    "lanl": ("#CC3311", "#E8836A"),
    "flame": ("#117733", "#6DB58A"),
}
EXTRA_COLORS = [("#882255", "#C27BA0"), ("#44AA99", "#93D1C6"), ("#332288", "#8C84C4")]


def paper_style():
    """Context manager applying PAPER_RC to the figures drawn inside it."""
    return plt.rc_context(PAPER_RC)


def pot_tex(n_pot: float) -> str:
    """An exposure in mathtext: 1e20 -> '$10^{20}$', 3.8e20 -> '$3.8\\times10^{20}$'."""
    exp = int(math.floor(math.log10(n_pot)))
    mant = n_pot / 10 ** exp
    if abs(mant - 1.0) < 1e-9:
        return f"$10^{{{exp}}}$"
    return f"${mant:.2g}\\times10^{{{exp}}}$"


def log_decades(ax, axis: str = "x", every: int = 1, skip_low: bool = False) -> None:
    """Major ticks at every decade, labelled 10^n (1 at 10^0) on every
    `every`-th decade; minor ticks at 2..9 x 10^n, unlabelled. skip_low drops
    the label at the axis' lower end, where it would collide with the other
    axis' first label in the corner."""
    target = ax.xaxis if axis == "x" else ax.yaxis

    def fmt(value, _pos):
        if value <= 0:
            return ""
        k = int(round(math.log10(value)))
        if abs(value - 10.0 ** k) > 1e-9 * value or k % every:
            return ""
        if skip_low and abs(value / min(target.get_view_interval()) - 1) < 1e-6:
            return ""
        return "$1$" if k == 0 else f"$10^{{{k}}}$"

    target.set_major_locator(mticker.LogLocator(base=10, numticks=60))
    target.set_major_formatter(mticker.FuncFormatter(fmt))
    target.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=60))
    target.set_minor_formatter(mticker.NullFormatter())


def family_colors(family: str, used: set) -> tuple:
    """(solid, light) for an experiment family, never repeating one already
    used on the same figure."""
    pair = FAMILY_COLORS.get(family)
    if pair is None or pair in used:
        pair = next((p for p in EXTRA_COLORS if p not in used), EXTRA_COLORS[0])
    used.add(pair)
    return pair


def save(fig, output):
    """Save as PDF (the given path) and a PNG next to it, like the paper."""
    from pathlib import Path
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output
