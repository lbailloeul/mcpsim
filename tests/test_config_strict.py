"""Config strictness: typos raise, silent fallbacks are gone."""

import pytest

from mcpsim.config import Config, DataConfig, TargetConfig, load_config


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_unknown_detector_key_raises(tmp_path):
    p = _write(tmp_path, "name: t\ndetector: {radius: 1.0}\n")
    with pytest.raises(ValueError, match="unknown key.*detector.*radius_m"):
        load_config(p)


def test_unknown_target_material_raises():
    with pytest.raises(ValueError, match="unknown target material 'lead'"):
        TargetConfig.from_dict({"name": "lead"})


def test_custom_target_with_full_params_ok():
    t = TargetConfig.from_dict({"name": "lead", "Z": 82, "A": 207.2,
                                "density_g_cm3": 11.35,
                                "interaction_length_cm": 17.6})
    assert t.Z == 82 and t.A == 207.2


def test_beam_energy_tolerance():
    assert Config().c_meson_rates()                        # 120 exact
    from dataclasses import replace
    cfg = Config()
    cfg = replace(cfg, beam=replace(cfg.beam, energy_gev=121.0))
    assert cfg.c_meson_rates()                             # within 2%
    cfg = replace(cfg, beam=replace(cfg.beam, energy_gev=130.0))
    with pytest.raises(ValueError, match="No tabulated meson multiplicities"):
        cfg.c_meson_rates()


def test_explicit_preset_typo_raises(tmp_path):
    p = _write(tmp_path, "name: t\npreset: darkqest\n")
    with pytest.raises(ValueError, match="unknown preset 'darkqest'"):
        load_config(p)


def test_inferred_preset_energy_mismatch_noted(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 60.0}\n")
    cfg = load_config(p)
    assert any("60 GeV" in n and "darkquest" in n for n in cfg.notes)


def test_detector_override_flips_geometry_flag(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                         "detector: {distance_m: 60.0}\n")
    assert load_config(p).geometry_is_preset is False
    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n")
    assert load_config(p2).geometry_is_preset is True


def test_needs_source_repo_respects_channels_and_abs_paths(tmp_path):
    cfg = Config(channels=["meson_decay"],
                 data=DataConfig(mass_grid=str(tmp_path / "g.txt"),
                                 acceptance={"pi0": str(tmp_path / "a.txt")}))
    assert cfg.needs_source_repo is False
    cfg2 = Config(channels=["meson_decay"],
                  data=DataConfig(mass_grid=str(tmp_path / "g.txt"),
                                  acceptance={"pi0": "relative.txt"}))
    assert cfg2.needs_source_repo is True
    # inherited brem/DY refs are irrelevant when those channels are off
    cfg3 = Config(channels=["meson_decay"],
                  data=DataConfig(mass_grid=str(tmp_path / "g.txt"),
                                  acceptance={}, brem_dir="SomeDir"))
    assert cfg3.needs_source_repo is False


def test_engine_config_validation():
    from mcpsim.config import EngineConfig
    with pytest.raises(ValueError, match="engine.decay"):
        EngineConfig(decay="fortran")
    with pytest.raises(ValueError, match="engine.fidelity"):
        EngineConfig(fidelity="fast")


def test_scintillator_derives_ngamma(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                         "detector: {type: bar_array, bar_length_m: 0.75}\n"
                         "sensitivity: {scintillator: plastic}\n")
    cfg = load_config(p)
    # plastic anchor: 2.5e5 @ 1.5 m -> half the length, half the light
    assert cfg.sensitivity.n_gamma == pytest.approx(1.25e5)

    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {type: bar_array, bar_length_m: 1.5}\n"
                          "sensitivity: {scintillator: cebr}\n")
    assert load_config(p2).sensitivity.n_gamma == pytest.approx(5.0e6)
    # linear scaling: a 0.5 m CeBr bar carries a third of the light
    p3 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {type: bar_array, bar_length_m: 0.5}\n"
                          "sensitivity: {scintillator: cebr}\n")
    assert load_config(p3).sensitivity.n_gamma == pytest.approx(5.0e6 / 3.0)


def test_scintillator_and_ngamma_together_raise(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                         "detector: {bar_length_m: 1.5}\n"
                         "sensitivity: {scintillator: plastic, n_gamma: 1.0e5}\n")
    with pytest.raises(ValueError, match="not both"):
        load_config(p)


def test_unknown_scintillator_raises(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                         "detector: {bar_length_m: 1.0}\n"
                         "sensitivity: {scintillator: nai}\n")
    with pytest.raises(ValueError, match="unknown scintillator 'nai'"):
        load_config(p)


def test_scintillator_needs_bar_length(tmp_path):
    # lanl_12bar carries no bar_length_m of its own, so there is nothing to
    # inherit and the derivation has no length to scale by.
    p = _write(tmp_path, "name: t\npreset: lanl_12bar\n"
                         "sensitivity: {scintillator: plastic}\n")
    with pytest.raises(ValueError, match="bar_length_m"):
        load_config(p)


def test_scintillator_inherits_bar_length_from_base_preset(tmp_path):
    # A 120 GeV config infers the darkquest base, which sets bar_length_m = 1.5,
    # so the derivation succeeds off the inherited length rather than erroring.
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                         "sensitivity: {scintillator: plastic}\n")
    assert load_config(p).sensitivity.n_gamma == pytest.approx(2.5e5)

    # An explicit length still wins over the inherited one.
    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {bar_length_m: 0.75}\n"
                          "sensitivity: {scintillator: plastic}\n")
    assert load_config(p2).sensitivity.n_gamma == pytest.approx(1.25e5)
