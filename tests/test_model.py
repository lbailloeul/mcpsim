"""compute_production assembly, warnings, and geometry honesty (mini repo)."""

from dataclasses import replace

import numpy as np
import pytest

from mcpsim.model import compute_production


def test_all_channels_assemble(mini_config):
    result = compute_production(mini_config)
    assert set(result.meson_channels) >= {"pi0", "eta", "rho", "omega", "phi", "jpsi"}
    assert result.brem_central is not None
    assert result.drell_yan is not None
    assert np.all(result.total >= 0) and np.any(result.total > 0)


def test_collider_mode_disables_brem_and_dy(mini_config):
    cfg = replace(mini_config, beam=replace(mini_config.beam, frame="collider"))
    result = compute_production(cfg)
    assert result.brem_central is None
    assert result.drell_yan is None
    assert sum("disabled" in w for w in result.warnings) == 2


def test_upsilon_skipped_for_unknown_family(mini_config):
    cfg = replace(mini_config, family="custom")
    result = compute_production(cfg)
    assert "upsilon" not in result.meson_channels
    assert any("upsilon" in w for w in result.warnings)


def test_upsilon_data_override(mini_config):
    cfg = replace(mini_config, family="custom",
                  data=replace(mini_config.data, upsilon_ageo=1e-4))
    result = compute_production(cfg)
    assert "upsilon" in result.meson_channels


def test_missing_mass_grid_actionable_error(mini_config, monkeypatch):
    monkeypatch.setenv("MCPSIM_SOURCE_REPO", "/nonexistent-mini")
    with pytest.raises(FileNotFoundError, match="MCPSIM_SOURCE_REPO"):
        compute_production(mini_config)


def test_overridden_geometry_without_samples_warns(mini_config):
    # geometry no longer the preset's; no archived samples in the mini repo
    cfg = replace(mini_config, geometry_is_preset=False)
    result = compute_production(cfg)
    assert any("PRESET-geometry file" in w for w in result.warnings)
    # falls back to the file, so the channel still exists
    assert "pi0" in result.meson_channels


def test_supplied_acceptance_file_is_used_as_it_is(mini_repo, tmp_path, stub_recompute,
                                                   flat_acceptance):
    import yaml
    from mcpsim.config import load_config
    # darkquest models the 1 x 1 m face, which its archived files are not for
    yields = {}
    for value in (0.4, 0.2):
        path = tmp_path / f"own_{value}.yaml"
        path.write_text(yaml.safe_dump({"preset": "darkquest", "channels": ["mesons"], "data": {
            "acceptance": {"pi0": str(flat_acceptance(f"pi0_{value}.txt", value))}}}))
        yields[value] = compute_production(load_config(path)).meson_channels
    assert np.any(yields[0.4]["pi0"] > 0)
    np.testing.assert_allclose(yields[0.4]["pi0"], 2 * yields[0.2]["pi0"], rtol=1e-12)
    np.testing.assert_allclose(yields[0.4]["eta"], yields[0.2]["eta"], rtol=1e-12)


def test_short_single_column_acceptance_raises(mini_config, tmp_path):
    acc = tmp_path / "one_col.txt"
    np.savetxt(acc, np.array([1e-3, 2e-3, 3e-3]))   # 3 rows, 1 col; the grid has 5
    data = replace(mini_config.data, acceptance={**mini_config.data.acceptance, "pi0": str(acc)})
    with pytest.raises(ValueError, match="single-column acceptance"):
        compute_production(replace(mini_config, channels=["meson_decay"], data=data))


def test_config_notes_reach_warnings(mini_config):
    cfg = replace(mini_config, notes=["synthetic note"])
    result = compute_production(cfg)
    assert "synthetic note" in result.warnings


@pytest.mark.parametrize("family", ["darkquest", "flame"])
def test_upsilon_borrow_is_reported(mini_config, family):
    # the upsilon is included on the J/psi's acceptance, and the run says so
    # rather than only when the channel is skipped
    result = compute_production(replace(mini_config, family=family))
    assert "upsilon" in result.meson_channels
    assert any("upsilon has no acceptance scan" in w for w in result.warnings)


def test_drell_yan_custom_geometry_warns(mini_config):
    from mcpsim import load_preset
    # the tables carry the preset's 1 x 1 m face: another detector is warned
    result = compute_production(mini_config)                 # r = 0.5 m disc
    assert result.drell_yan is not None
    assert any("LO acceptance of the preset's detector (1 x 1 m face at 40 m)" in w
               for w in result.warnings)
    face = replace(mini_config, detector=load_preset("darkquest").detector)
    assert not any("acceptance" in w and "Drell-Yan" in w
                   for w in compute_production(face).warnings)
    # a scan's acceptance file is the preset geometry's
    scan = replace(mini_config, dy_cross_is_preset=False, geometry_is_preset=False)
    assert any("Drell-Yan acceptance is the PRESET-geometry file" in w
               for w in compute_production(scan).warnings)


def test_drell_yan_normalisation_follows_target(mini_config):
    # A user-supplied scan (data.dy_cross set in the config) is used as is,
    # with the per-POT normalisation of the configured target.
    from mcpsim.config import TargetConfig
    from mcpsim.physics.constants import SIGMA_PN_INELASTIC_MB
    from mcpsim.physics.drellyan import sigma_inel_per_nucleon_mb
    user_scan = replace(mini_config, dy_cross_is_preset=False)
    legacy = compute_production(replace(user_scan, engine=replace(user_scan.engine, fidelity="legacy")))
    for name in ("iron", "molybdenum"):
        t = TargetConfig().overlay({"name": name})
        cfg = replace(user_scan, target=t, data=replace(user_scan.data, dy_material="iron"))
        res = compute_production(cfg)
        ratio = SIGMA_PN_INELASTIC_MB / sigma_inel_per_nucleon_mb(t.density_g_cm3, t.interaction_length_cm)
        np.testing.assert_allclose(res.drell_yan, legacy.drell_yan * ratio, rtol=1e-12)
        mismatch = [w for w in res.warnings if "simulated with iron nuclear PDFs" in w]
        assert bool(mismatch) == (name != "iron")


def test_drell_yan_undeclared_material_warns(mini_config):
    res = compute_production(replace(mini_config, dy_cross_is_preset=False))  # no dy_material
    assert any("do not declare the material" in w for w in res.warnings)


@pytest.mark.parametrize("name", ["iron", "molybdenum", "tungsten", "carbon"])
def test_drell_yan_uses_the_target_material_table(mini_config, mini_repo, name):
    from mcpsim import paths
    from mcpsim.config import TargetConfig
    from mcpsim.physics import drellyan
    from mcpsim.physics.constants import MB_TO_PB
    t = TargetConfig().overlay({"name": name})
    res = compute_production(replace(mini_config, target=t))
    table = drellyan.load_table(drellyan.find_table(paths.DY_TABLES_DIR, 120.0, name))
    assert table.header["material"] == name and table.header["beam_energy_gev"] == "120"
    per_mass = (mini_config.beam.n_pot * table.sigma_pb * table.acc
                / (drellyan.sigma_inel_per_nucleon_mb(t.density_g_cm3, t.interaction_length_cm) * MB_TO_PB))
    np.testing.assert_allclose(res.drell_yan, np.interp(res.masses, table.masses, per_mass),
                               rtol=1e-12)
    assert not any("Drell-Yan cross sections were simulated" in w or "no built-in" in w
                   for w in res.warnings)


def test_drell_yan_tables_differ_by_material(mini_config):
    from mcpsim.config import TargetConfig
    from mcpsim.physics.drellyan import sigma_inel_per_nucleon_mb
    # undo the per-POT normalisation: what is left is the material's table
    per_nucleon = {}
    for name in ("carbon", "iron", "molybdenum", "tungsten"):
        t = TargetConfig().overlay({"name": name})
        dy = compute_production(replace(mini_config, target=t)).drell_yan
        per_nucleon[name] = dy[-1] * sigma_inel_per_nucleon_mb(t.density_g_cm3, t.interaction_length_cm)
    values = sorted(per_nucleon.values())
    assert all(b / a > 1.001 for a, b in zip(values, values[1:]))     # all distinct
    assert per_nucleon["carbon"] == values[-1]


def test_drell_yan_legacy_keeps_the_paper_scan(mini_config):
    from mcpsim.config import TargetConfig
    from mcpsim.physics.constants import MB_TO_PB, SIGMA_PN_INELASTIC_MB
    legacy = replace(mini_config, engine=replace(mini_config.engine, fidelity="legacy"),
                     data=replace(mini_config.data, dy_material="iron"))
    res = compute_production(legacy)
    expected = mini_config.beam.n_pot * 2.0 / (SIGMA_PN_INELASTIC_MB * MB_TO_PB) * 0.01
    np.testing.assert_allclose(res.drell_yan, expected, rtol=1e-12)   # dy_cross.txt: 2 pb
    assert not any("Drell-Yan" in w for w in res.warnings)
    # SHiP's published setup: the iron scan and 13 mb on molybdenum, said as such
    res = compute_production(replace(legacy, target=TargetConfig().overlay({"name": "molybdenum"})))
    np.testing.assert_allclose(res.drell_yan, expected, rtol=1e-12)
    assert any("fidelity legacy, as published" in w for w in res.warnings)


def test_drell_yan_without_table_falls_back_to_the_scan(mini_config):
    cfg = replace(mini_config, beam=replace(mini_config.beam, energy_gev=200.0),
                  channels=["drell_yan"])          # no meson multiplicities at 200 GeV
    res = compute_production(cfg)
    assert res.drell_yan is not None and res.drell_yan.max() > 0
    assert any("no built-in Drell-Yan table for target 'iron' at 200 GeV" in w for w in res.warnings)
    # ... and with neither a table nor a scan the channel is skipped, loudly.
    res = compute_production(replace(cfg, data=replace(cfg.data, dy_cross=None)))
    assert res.drell_yan is None
    assert any("no Drell-Yan cross sections" in w for w in res.warnings)
