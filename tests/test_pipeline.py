"""regenerate_inputs: warning propagation, guard ordering, early skips."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from mcpsim import load_preset
from mcpsim.generation.common import ToolchainError
from mcpsim.pipeline import regenerate_inputs


def test_the_engine_seed_reaches_the_cpp_binaries(mini_repo, tmp_path, monkeypatch):
    from mcpsim.generation import common
    seeds = {}

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        tool = Path(cmd[0]).name
        if tool == "generatePion_12bar":       # out trials seed mother_MeV
            seeds.setdefault(tool, set()).add(cmd[3])
            Path(cmd[1]).write_text("1000.0 0.999 0.0\n")
        elif tool == "lanl_decayPion_12bar":   # root mchi eff source distance cols rows mother seed ...
            seeds.setdefault(tool, set()).add(cmd[9])
            Path(cmd[3]).write_text(f"{cmd[2]} 0.25\n")

    monkeypatch.setattr(common, "compile_cpp", lambda src, out, *a, **k: out)
    monkeypatch.setattr(common, "run", fake_run)
    lanl = load_preset("lanl_12bar")
    cfg = replace(lanl, engine=replace(lanl.engine, decay="cpp", seed=777))
    regenerate_inputs(cfg, tmp_path / "gen", mass_subset=2)
    assert seeds == {"generatePion_12bar": {"777"}, "lanl_decayPion_12bar": {"777"}}


def test_regenerating_with_a_missing_mass_grid_names_the_grid(mini_config, tmp_path):
    import pytest
    cfg = replace(mini_config, data=replace(mini_config.data, mass_grid="no_such_grid.txt"))
    with pytest.raises(FileNotFoundError, match="mass grid 'no_such_grid.txt' not found"):
        regenerate_inputs(cfg, tmp_path / "gen")


def test_local_drell_yan_runs_madgraph_on_the_cores_asked_for(mini_config, tmp_path, monkeypatch):
    from mcpsim.generation import common, madgraph
    asked = []

    def fake_madgraph(cfg, model, mchi, proc_dir, *, nevents, cores=None):
        asked.append(cores)
        lhe = proc_dir / "unweighted_events.lhe"
        lhe.parent.mkdir(parents=True, exist_ok=True)
        pdg = madgraph.MCP_PDG_DEFAULT
        lhe.write_text(f"""<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 6.0e+01 6.0e+01 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+01 1.0e+01 0 0 -1
 {pdg} 1 1 2 0 0 0.0 0.0 5.0e+01 5.0e+01 1.0e-01 0 1
-{pdg} 1 1 2 0 0 0.0 0.0 1.0e+00 1.0e+00 1.0e-01 0 -1
</event>
""")
        return 2.0, lhe

    monkeypatch.setattr(common, "require_command", lambda *a, **k: "mg5_aMC")
    monkeypatch.setattr(madgraph, "_run_single_mass", fake_madgraph)
    cfg = replace(mini_config, channels=["drell_yan"])
    regenerate_inputs(cfg, tmp_path / "six", mass_subset=2, gen_opts={"n_proc": 6})
    regenerate_inputs(cfg, tmp_path / "unset", mass_subset=2)
    assert asked == [6, 6, None, None]


def test_regenerated_acceptance_is_used_by_the_next_compute(mini_repo, tmp_path, monkeypatch,
                                                            stub_recompute):
    from mcpsim.generation import pydecay
    from mcpsim.model import compute_production
    monkeypatch.setattr(pydecay, "acceptance_on_grid",
                        lambda cfg, name, grid, **k: np.full(grid.size, 0.3))
    cfg = replace(load_preset("darkquest"), channels=["meson_decay"])
    new_cfg, _ = regenerate_inputs(cfg, tmp_path / "gen")
    before, after = (compute_production(c).meson_channels["pi0"] for c in (cfg, new_cfg))
    assert np.any(before > 0)
    np.testing.assert_allclose(after, before * 0.3 / stub_recompute, rtol=1e-12)


def test_a_regenerated_drell_yan_acceptance_is_this_detectors(mini_repo, tmp_path, monkeypatch):
    from mcpsim.generation import madgraph
    from mcpsim.model import compute_production
    masses = np.loadtxt(mini_repo / "data-backup" / "mship_values.txt")
    files = (tmp_path / "dy_cross.txt", tmp_path / "ageo_dy.txt")
    np.savetxt(files[0], np.full(masses.size, 2.0))
    np.savetxt(files[1], np.column_stack([masses, np.full(masses.size, 0.01)]))
    monkeypatch.setattr(madgraph, "generate_drell_yan", lambda *a, **k: files)
    monkeypatch.setattr(madgraph, "prepare_condor_dy", lambda *a, **k: files)
    cfg = replace(load_preset("darkquest"), channels=["drell_yan"])
    for opts in (None, {"condor": True, "submit": False}):
        new_cfg, _ = regenerate_inputs(cfg, tmp_path / "gen", gen_opts=opts)
        result = compute_production(new_cfg)
        assert result.drell_yan.max() > 0
        assert [w for w in result.warnings if "Drell-Yan" in w] == []


def test_toolchain_failures_return_warnings(mini_config, tmp_path, monkeypatch):
    """cpp engine with every backend unavailable: nothing crashes, every skip
    is returned as a warning (not just printed)."""
    from mcpsim.generation import brem_run, madgraph, mesons

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
    written = write_tables(result, tmp_path)
    # Already above threshold at charge_min: the table value is a floor.
    assert any("bottom of the charge scan" in w for w in result.warnings)
    record = tmp_path / f"run_{cfg.tag}.yaml"
    assert record in written
    rec = yaml.safe_load(record.read_text())
    assert rec["config"]["name"] == cfg.name and rec["config"]["sensitivity"]["charge_min"] == 0.5
    assert "mcpsim_version" in rec
    # the Drell-Yan cross sections actually used: the iron table, not dy_cross
    assert rec["drell_yan_cross_sections"].endswith("dy_sigma_iron_120GeV.txt")


def test_the_limit_table_and_the_limit_plot_show_the_same_limit(mini_config, tmp_path):
    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    from mcpsim.plotting.limits import limit_curves
    sens = replace(mini_config.sensitivity, n_chi_threshold=10.0, n_charge=50)
    for cfg in (mini_config, replace(mini_config, sensitivity=sens)):
        result = compute_production(cfg)
        write_tables(result, tmp_path)
        table = np.loadtxt(tmp_path / f"limit_{cfg.tag}.txt")
        np.testing.assert_allclose(table[:, 1], limit_curves(result)[0], rtol=1e-7)


def test_a_result_carries_the_detector_its_run_modelled(mini_config, tmp_path):
    from mcpsim.config import DetectorConfig
    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    paper = DetectorConfig(type="cylindrical", radius_m=0.3, distance_m=40.0)
    cfg = replace(mini_config, legacy_detector=paper,
                  engine=replace(mini_config.engine, fidelity="legacy"))
    result = compute_production(cfg)
    assert result.config.detector == paper
    write_tables(result, tmp_path)
    assert "Detector: r = 0.3 m disc at 40 m" in (tmp_path / f"limit_curve_{cfg.tag}.txt").read_text()


def test_the_limit_curve_lists_only_the_channels_that_gave_a_yield(mini_config, tmp_path):
    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    result = compute_production(replace(mini_config, channels=["meson_decay", "brem"]))
    left_out = replace(result, brem_central=None, brem_low=None, brem_high=None)
    write_tables(left_out, tmp_path)
    text = (tmp_path / f"limit_curve_{mini_config.tag}.txt").read_text()
    assert "Channels: meson_decay (fidelity corrected)" in text


def test_limit_curve_file_for_collaborators(mini_config, tmp_path):
    from mcpsim.model import compute_production
    from mcpsim.pipeline import write_tables
    result = compute_production(mini_config)
    written = write_tables(result, tmp_path)
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
