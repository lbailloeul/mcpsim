"""regenerate_inputs: warning propagation, guard ordering, early skips."""

from dataclasses import replace

import numpy as np
import pytest

from mcpsim.generation.common import ToolchainError
from mcpsim.pipeline import regenerate_inputs


def test_toolchain_failures_return_warnings(mini_config, tmp_path, monkeypatch):
    """cpp engine with every backend unavailable: nothing crashes, every skip
    is returned as a warning (not just printed)."""
    from mcpsim.generation import brem_run, decay, madgraph, mesons

    cfg = replace(mini_config,
                  engine=replace(mini_config.engine, decay="cpp"))
    monkeypatch.setattr(mesons, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(ToolchainError("no PYTHIA")))
    monkeypatch.setattr(brem_run, "regenerate_brem",
                        lambda *a, **k: (_ for _ in ()).throw(ToolchainError("no mpirun")))
    monkeypatch.setattr(madgraph, "generate_drell_yan",
                        lambda *a, **k: (_ for _ in ()).throw(ToolchainError("no mg5")))

    new_cfg, warnings = regenerate_inputs(cfg, tmp_path / "gen")
    assert any("no mpirun" in w for w in warnings)
    assert any("no mg5" in w for w in warnings)
    assert len(warnings) >= 3  # mesons (per meson or geometry guard) + brem + dy


def test_regenerated_drell_yan_scan_is_used_as_it_is(mini_config, tmp_path, monkeypatch):
    """A fresh MadGraph scan replaces the built-in table (cross sections and
    the acceptance of the configured detector), locally or from a batch."""
    from mcpsim.generation import madgraph
    from mcpsim.model import drell_yan_table

    cfg = replace(mini_config, channels=["drell_yan"])
    assert drell_yan_table(cfg) is not None                  # the iron table, by default
    files = (tmp_path / "dy_cross_mini.txt", tmp_path / "ageo_dy_mini.txt")
    monkeypatch.setattr(madgraph, "generate_drell_yan", lambda *a, **k: files)
    monkeypatch.setattr(madgraph, "prepare_condor_dy", lambda *a, **k: files)
    for opts in (None, {"condor": True, "submit": False}):
        new_cfg, warnings = regenerate_inputs(cfg, tmp_path / "gen", gen_opts=opts)
        assert new_cfg.data.dy_cross == str(files[0]) and new_cfg.data.dy_material == "iron"
        assert not new_cfg.dy_cross_is_preset and drell_yan_table(new_cfg) is None
        assert any("HTCondor batch" in w for w in warnings) == bool(opts)

    # a failed regeneration keeps the precomputed inputs, table included
    monkeypatch.setattr(madgraph, "generate_drell_yan",
                        lambda *a, **k: (_ for _ in ()).throw(ToolchainError("no mg5")))
    new_cfg, _ = regenerate_inputs(cfg, tmp_path / "gen")
    assert new_cfg.dy_cross_is_preset and drell_yan_table(new_cfg) is not None


def test_geometry_guard_runs_before_meson_generation(mini_config, tmp_path, monkeypatch):
    """check_supported must fire before the expensive generation step.

    An off-axis detector is unsupported by the cpp engine for EVERY meson, so
    no generation may run at all."""
    from mcpsim.generation import mesons

    calls = []
    monkeypatch.setattr(mesons, "generate",
                        lambda *a, **k: calls.append("generate") or None)
    cfg = replace(mini_config,
                  detector=replace(mini_config.detector, offaxis_mrad=4.0),
                  engine=replace(mini_config.engine, decay="cpp"),
                  channels=["meson_decay"])
    _, warnings = regenerate_inputs(cfg, tmp_path / "gen")
    assert calls == [], "meson generation ran before the geometry guard"
    assert any("regeneration skipped" in w for w in warnings)


def test_untabulated_energy_skips_meson_channel_early(mini_config, tmp_path, monkeypatch):
    from mcpsim.generation import pydecay

    ran = []
    monkeypatch.setattr(pydecay, "acceptance_on_grid",
                        lambda *a, **k: ran.append(1) or None)
    cfg = replace(mini_config,
                  beam=replace(mini_config.beam, energy_gev=30.0),
                  channels=["meson_decay"])
    _, warnings = regenerate_inputs(cfg, tmp_path / "gen")
    assert ran == [], "acceptance work ran despite missing multiplicities"
    assert any("meson_decay regeneration skipped" in w for w in warnings)


def test_limit_floor_warning_and_run_record(mini_config, tmp_path):
    import yaml
    from dataclasses import replace

    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    sens = replace(mini_config.sensitivity, charge_min=0.5, charge_max=1.0, n_charge=10)
    cfg = replace(mini_config, sensitivity=sens)
    result = compute_production(cfg)
    written = write_tables(cfg, result, tmp_path)
    # Already above threshold at charge_min: the table value is a floor.
    assert any("bottom of the charge scan" in w for w in result.warnings)
    record = tmp_path / f"run_{cfg.tag}.yaml"
    assert record in written
    rec = yaml.safe_load(record.read_text())
    assert rec["config"]["name"] == cfg.name and rec["config"]["sensitivity"]["charge_min"] == 0.5
    assert "mcpsim_version" in rec
    # the Drell-Yan cross sections actually used: the iron table, not dy_cross
    assert rec["drell_yan_cross_sections"].endswith("dy_sigma_iron_120GeV.txt")


def test_limit_curve_file_for_collaborators(mini_config, tmp_path):
    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    result = compute_production(mini_config)
    written = write_tables(mini_config, result, tmp_path)
    curve = tmp_path / f"limit_curve_{mini_config.tag}.txt"
    assert curve in written
    text = curve.read_text()
    for needed in ("excluded region lies above", "N = 3.09", "n_gamma", "Beam: 120 GeV protons on iron",
                   "Detector: r = 0.5 m disc at 40 m", "fidelity corrected", "m_chi [GeV]"):
        assert needed in text
    table = np.loadtxt(tmp_path / f"limit_{mini_config.tag}.txt")
    data = np.loadtxt(curve)                      # loads as it is: 2 clean columns
    finite = np.isfinite(table[:, 1])
    np.testing.assert_allclose(data, table[finite], rtol=1e-6)
