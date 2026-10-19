"""The paper-style plotting layer: exposure typesetting, endpoints, and that
every figure writes its PDF + PNG (mini repo, no real data needed)."""

from dataclasses import replace

import numpy as np

from mcpsim.plotting.style import family_colors, pot_tex


def test_pot_tex():
    assert pot_tex(1e20) == "$10^{20}$"
    assert pot_tex(2e20) == r"$2\times10^{20}$"
    assert pot_tex(3.8e20) == r"$3.8\times10^{20}$"
    assert pot_tex(2e22) == r"$2\times10^{22}$"
    assert pot_tex(9.96e20) == "$10^{21}$"          # two figures round up a decade
    assert pot_tex(1.04e20) == "$10^{20}$"


def test_endpoint_keeps_the_drop_and_blanks_the_start():
    from mcpsim.plotting.production import _endpoint
    y = _endpoint(np.array([0.0, 0.0, 3.0, 2.0, 0.0, 0.0]))
    assert np.isnan(y[:2]).all()                     # no line before the channel opens
    np.testing.assert_array_equal(y[2:], [3.0, 2.0, 0.0, 0.0])   # zeros draw the drop
    assert np.isnan(_endpoint(np.zeros(3))).all()


def test_family_colors_never_repeat():
    used = set()
    first, second = family_colors("darkquest", used), family_colors("darkquest", used)
    assert first == ("#FF7F0E", "#FFAC62") and second != first


def test_figures_write_pdf_and_png(mini_config, tmp_path):
    from mcpsim.model import compute_production
    from mcpsim.plotting.limits import plot_limit, plot_limit_compare, plot_limit_overlay
    from mcpsim.plotting.production import plot_production, plot_production_overlay
    res = compute_production(mini_config)
    paths = [plot_production(res, tmp_path / "production.pdf"),
             plot_limit(res, tmp_path / "limit.pdf"),
             plot_limit_compare([res, replace(res, config=replace(res.config, name="mini 2"))],
                                tmp_path / "compare.pdf")]
    scan = [replace(res, config=replace(res.config, name=f"mini {a}",
                                        detector=replace(res.config.detector, offaxis_mrad=a)))
            for a in (0.0, 2.0)]
    paths += [plot_production_overlay(mini_config, scan, tmp_path),
              plot_limit_overlay(mini_config, scan, tmp_path)]
    for p in paths:
        assert p.exists() and p.with_suffix(".png").exists(), p


def test_both_plots_name_a_run_the_same_way(mini_config, monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from mcpsim.model import compute_production
    from mcpsim.plotting import limits, production
    figures = []
    for module in (limits, production):
        monkeypatch.setattr(module, "save", lambda fig, output: figures.append(fig) or output)
    legacy = replace(mini_config, engine=replace(mini_config.engine, fidelity="legacy"))
    for cfg, shown in ((mini_config, "mini"), (legacy, "mini (legacy)"),
                       (replace(legacy, name="mini legacy"), "mini legacy")):   # the CLI's name
        figures.clear()
        res = compute_production(cfg)
        production.plot_production(res, tmp_path / "production.pdf")
        limits.plot_limit(res, tmp_path / "limit.pdf")
        title = figures[0].axes[0].get_legend().get_title().get_text()
        labels = [t.get_text() for t in figures[1].axes[0].get_legend().get_texts()]
        plt.close("all")
        assert title.splitlines()[0] == shown
        assert labels and all(label.startswith(f"{shown}: ") for label in labels), labels


def test_limit_curves_without_brem(mini_config):
    from mcpsim.model import compute_production
    from mcpsim.plotting.limits import limit_curves
    res = compute_production(mini_config)
    eps, no_brem = limit_curves(res)
    assert no_brem is not None
    ok = np.isfinite(eps) & np.isfinite(no_brem)
    assert np.all(eps[ok] <= no_brem[ok])            # brem only adds signal
    res.brem_central = None
    assert limit_curves(res)[1] is None
