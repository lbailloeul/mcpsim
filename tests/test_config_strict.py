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


def test_user_dy_cross_replaces_the_builtin_tables(tmp_path):
    # Your own Drell-Yan scan wins over the per-material tables; a preset's
    # scan (or a target swap) keeps the tables in force.
    p = _write(tmp_path, "preset: darkquest\ndata: {dy_cross: my_scan.txt}\n")
    assert load_config(p).dy_cross_is_preset is False
    p2 = _write(tmp_path, "preset: darkquest\ntarget: {name: tungsten}\n")
    assert load_config(p2).dy_cross_is_preset is True


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

    # cebr anchor: 5.0e6 @ 1.5 m
    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {type: bar_array, bar_length_m: 1.5}\n"
                          "sensitivity: {scintillator: cebr}\n")
    assert load_config(p2).sensitivity.n_gamma == pytest.approx(5.0e6)
    # the LANSCE 12-bar demonstrator's 5 cm crystals scale down 30x
    p3 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {type: bar_array, bar_length_m: 0.05}\n"
                          "sensitivity: {scintillator: cebr}\n")
    assert load_config(p3).sensitivity.n_gamma == pytest.approx(5.0e6 / 30.0)


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
    # No beam energy -> no base preset is inferred, so there is no bar_length_m
    # to inherit and the derivation has no length to scale by.
    p = _write(tmp_path, "name: t\ndetector: {type: bar_array}\n"
                         "sensitivity: {scintillator: plastic}\n")
    with pytest.raises(ValueError, match="bar_length_m"):
        load_config(p)

    # Explicitly clearing an inherited length raises too.
    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\n"
                          "detector: {bar_length_m: null}\n"
                          "sensitivity: {scintillator: plastic}\n")
    with pytest.raises(ValueError, match="bar_length_m"):
        load_config(p2)


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


def test_unknown_top_level_key_raises_with_suggestion(tmp_path):
    # A misspelled section used to be dropped silently (preset geometry kept).
    p = _write(tmp_path, "name: t\ndetecor: {distance_m: 50.0}\n")
    with pytest.raises(ValueError, match="unknown top-level key.*did you mean 'detector'"):
        load_config(p)


def test_channels_validated_and_aliased(tmp_path):
    p = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\nchannels: [mesons, dy]\n")
    assert load_config(p).channels == ["meson_decay", "drell_yan"]
    p2 = _write(tmp_path, "name: t\nbeam: {energy_gev: 120.0}\nchannels: [brems]\n")
    with pytest.raises(ValueError, match="unknown channel 'brems'.*did you mean 'brem'"):
        load_config(p2)


def test_bar_length_override_rederives_inherited_scintillator(tmp_path):
    # FLAME derives n_gamma from plastic @ 1.5 m; a shorter bar must follow it.
    p = _write(tmp_path, "preset: flame\nname: t\ndetector: {bar_length_m: 0.75}\n")
    assert load_config(p).sensitivity.n_gamma == pytest.approx(1.25e5)
    # An explicit n_gamma still wins.
    p2 = _write(tmp_path, "preset: flame\nname: t\ndetector: {bar_length_m: 0.75}\n"
                          "sensitivity: {n_gamma: 3.0e5}\n")
    assert load_config(p2).sensitivity.n_gamma == pytest.approx(3.0e5)


def test_target_aliases_in_yaml(tmp_path):
    for alias, name in (("fe", "iron"), ("mo", "molybdenum"), ("tzm", "molybdenum"),
                        ("w", "tungsten"), ("graphite", "carbon")):
        p = _write(tmp_path, f"name: t\nbeam: {{energy_gev: 120.0}}\ntarget: {{name: {alias}}}\n")
        assert load_config(p).target.name == name


def test_legacy_fidelity_models_the_published_detector(tmp_path):
    from dataclasses import replace
    from mcpsim import geometry, load_preset
    from mcpsim.config import run_config
    from mcpsim.model import acceptance_files_valid
    cfg = load_preset("darkquest")
    legacy = replace(cfg, engine=replace(cfg.engine, fidelity="legacy"))
    # default: the 1 x 1 m face, for which the archived files are not valid
    assert geometry.face_signature(run_config(cfg).detector) == "1 x 1 m face at 40 m"
    assert not acceptance_files_valid(run_config(cfg))
    # legacy: the paper's disc and its files
    assert run_config(legacy).detector == cfg.legacy_detector
    assert acceptance_files_valid(run_config(legacy))
    assert run_config(run_config(legacy)) == run_config(legacy)          # idempotent
    # a config's own detector is kept even under legacy
    p = _write(tmp_path, "preset: darkquest\nengine: {fidelity: legacy}\n"
                         "detector: {distance_m: 60.0}\n")
    own = run_config(load_config(p))
    assert geometry.face_signature(own.detector) == "1 x 1 m face at 60 m"
    assert not acceptance_files_valid(own)
    # presets without a separate published detector are unaffected
    flame = load_preset("flame")
    assert flame.legacy_detector is None and acceptance_files_valid(flame)


def test_legacy_detector_is_strict(tmp_path):
    p = _write(tmp_path, "preset: darkquest\nlegacy_detector: {radius: 0.5}\n")
    with pytest.raises(ValueError, match="legacy_detector"):
        load_config(p)
