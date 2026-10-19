"""Config loading and strictness: presets and YAML numbers load as they
should, typos raise, silent fallbacks are gone, run records round-trip."""

from dataclasses import replace

import pytest
import yaml

from mcpsim import list_presets, load_preset
from mcpsim.config import (BeamConfig, Config, DataConfig, DetectorConfig, EngineConfig,
                           load_config, normalize_target)
from mcpsim.physics import constants as C


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def _load(tmp_path, raw):
    return load_config(_write(tmp_path, yaml.safe_dump(raw)))


def test_scientific_notation_without_sign_parses_as_float(tmp_path):
    # The crux of an earlier bug: PyYAML's SafeLoader reads '3.0e20' as a string.
    path = tmp_path / "exposure.yaml"
    path.write_text("preset: darkquest\nbeam: {n_pot: 3.0e20}\n"
                    "sensitivity: {n_gamma: 1.5e5, charge_min: 2e-5}\n")
    cfg = load_config(path)
    for value, wanted in ((cfg.beam.n_pot, 3e20), (cfg.sensitivity.n_gamma, 1.5e5),
                          (cfg.sensitivity.charge_min, 2e-5)):
        assert isinstance(value, float) and value == wanted


def test_all_presets_load_with_numeric_fields():
    for name in list_presets():
        cfg = load_preset(name)
        assert isinstance(cfg.beam.n_pot, float)
        assert isinstance(cfg.beam.energy_gev, float)
        assert isinstance(cfg.sensitivity.n_gamma, float)


def test_target_material_resolution():
    cfg = load_preset("ship")
    assert cfg.target.name == "molybdenum"
    assert cfg.target.A == 95.95
    assert cfg.target.density_g_cm3 == 10.2


def test_unknown_detector_key_raises(tmp_path):
    p = _write(tmp_path, "name: t\ndetector: {radius: 1.0}\n")
    with pytest.raises(ValueError, match="unknown key.*detector.*radius_m"):
        load_config(p)


@pytest.mark.parametrize("spec", [{"name": "lead"}, {"name": "Lead", "Z": 82, "A": 207.2}])
def test_unknown_target_material_raises(tmp_path, spec):
    with pytest.raises(ValueError, match="unknown target material"):
        _load(tmp_path, {"preset": "ship", "target": spec})


def test_custom_target_with_full_params_ok(tmp_path):
    t = _load(tmp_path, {"preset": "ship", "target": {
        "name": "lead", "Z": 82, "A": 207.2, "density_g_cm3": 11.35,
        "interaction_length_cm": 17.6}}).target
    assert (t.name, t.Z, t.A, t.density_g_cm3, t.interaction_length_cm) == (
        "lead", 82, 207.2, 11.35, 17.6)


def test_acceptance_overrides_merge_per_meson(tmp_path):
    cfg = _load(tmp_path, {"preset": "darkquest",
                           "data": {"acceptance": {"pi0": "/work/pi0.txt", "eta": None}}})
    expected = {m: f for m, f in load_preset("darkquest").data.acceptance.items() if m != "eta"}
    assert cfg.data.acceptance == {**expected, "pi0": "/work/pi0.txt"}


def test_acceptance_meson_typo_raises(tmp_path):
    with pytest.raises(ValueError, match="pio"):
        _load(tmp_path, {"preset": "darkquest", "data": {"acceptance": {"pio": "/work/pi0.txt"}}})


def test_partial_target_override_keeps_the_base_material(tmp_path):
    cfg = _load(tmp_path, {"preset": "ship", "target": {"density_g_cm3": 8.0}})
    assert cfg.target == replace(load_preset("ship").target, density_g_cm3=8.0)


@pytest.mark.parametrize("spelling, preset", [
    ("Iron", "darkquest"), (" IRON ", "darkquest"), ("fe", "darkquest"), ("FE", "darkquest"),
    ("Mo", "ship"), (" MO ", "ship"), ("tzm", "ship"), ("TZM", "ship"),
    ("w", "lanl_12bar"), ("Graphite", "flame")])
def test_yaml_target_names_normalise_like_the_target_flag(tmp_path, spelling, preset):
    # a base of another material, so the result can only come from the name
    base = "darkquest" if preset == "lanl_12bar" else "lanl_12bar"
    target = _load(tmp_path, {"preset": base, "target": {"name": spelling}}).target
    assert target == load_preset(preset).target
    assert target.name == normalize_target(spelling)


@pytest.mark.parametrize("energy, tabulated", [
    (120.0, "120"), (121.0, "120"), (400.0, "400"), (0.8, "0.8"), (130.0, None)])
def test_a_beam_gets_the_multiplicities_tabulated_for_its_energy(energy, tabulated):
    cfg = replace(Config(), beam=replace(Config().beam, energy_gev=energy))
    if tabulated is None:
        with pytest.raises(ValueError, match="No tabulated meson multiplicities"):
            cfg.c_meson_rates()
    else:
        assert cfg.c_meson_rates() == C.C_MESON_BY_BEAM[tabulated]


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


@pytest.mark.parametrize("section, key, typo, meant, valid", [
    ("detector", "type", "bar-array", "bar_array", ("cylindrical", "bar_array")),
    ("beam", "frame", "colider", "collider", ("fixed_target", "collider")),
    ("engine", "decay", "cp", "cpp", ("python", "cpp")),
    ("engine", "fidelity", "legasy", "legacy", ("corrected", "legacy")),
], ids=["detector.type", "beam.frame", "engine.decay", "engine.fidelity"])
def test_an_unknown_choice_raises_at_load_with_a_suggestion(tmp_path, section, key,
                                                            typo, meant, valid):
    with pytest.raises(ValueError) as err:
        _load(tmp_path, {"preset": "darkquest", section: {key: typo}})
    for part in (f"{section}.{key}", f"'{typo}'", f"did you mean '{meant}'", *valid):
        assert part in str(err.value)


@pytest.mark.parametrize("build", [lambda: DetectorConfig(type="bar-array"),
                                   lambda: BeamConfig(frame="colider"),
                                   lambda: EngineConfig(decay="cp"),
                                   lambda: EngineConfig(fidelity="legasy")],
                         ids=["detector.type", "beam.frame", "engine.decay", "engine.fidelity"])
def test_an_unknown_choice_cannot_be_built(build):
    with pytest.raises(ValueError, match="did you mean"):
        build()


def test_only_the_rules_table_tells_the_fidelity_modes_apart():
    import re
    from mcpsim import paths
    mode = r"[\"'](?:legacy|corrected)[\"']"
    compares_a_mode = re.compile(rf"[=!]=\s*{mode}|{mode}\s*[=!]=")
    found = [f"{path.relative_to(paths.PACKAGE_DIR)}:{n}: {line.strip()}"
             for path in sorted(paths.PACKAGE_DIR.rglob("*.py"))
             if path.relative_to(paths.PACKAGE_DIR).parts[0] != "legacy"
             for n, line in enumerate(path.read_text().splitlines(), 1)
             if compares_a_mode.search(line)]
    assert found == [], "read cfg.engine.rules instead:\n" + "\n".join(found)


def test_an_upsilon_acceptance_and_sample_cannot_both_be_given(tmp_path):
    with pytest.raises(ValueError, match="upsilon_ageo and data.upsilon_sample"):
        _load(tmp_path, {"preset": "flame",
                         "data": {"upsilon_ageo": 1e-4, "upsilon_sample": "upsilon.root"}})


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


def test_legacy_fidelity_models_the_published_detector(tmp_path):
    from mcpsim import geometry
    from mcpsim.config import run_config
    from mcpsim.model import acceptance_file_valid
    cfg = load_preset("darkquest")
    preset_file = cfg.data.acceptance["pi0"]
    legacy = replace(cfg, engine=replace(cfg.engine, fidelity="legacy"))
    # default: the 1 x 1 m face, for which the archived files are not valid
    assert geometry.face_signature(run_config(cfg).detector) == "1 x 1 m face at 40 m"
    assert not acceptance_file_valid(run_config(cfg), preset_file)
    # legacy: the paper's disc and its files
    assert run_config(legacy).detector == cfg.legacy_detector
    assert acceptance_file_valid(run_config(legacy), preset_file)
    assert run_config(run_config(legacy)) == run_config(legacy)          # idempotent
    # a config's own detector is kept even under legacy
    p = _write(tmp_path, "preset: darkquest\nengine: {fidelity: legacy}\n"
                         "detector: {distance_m: 60.0}\n")
    own = run_config(load_config(p))
    assert geometry.face_signature(own.detector) == "1 x 1 m face at 60 m"
    assert not acceptance_file_valid(own, preset_file)
    # presets without a separate published detector are unaffected
    flame = load_preset("flame")
    assert flame.legacy_detector is None and acceptance_file_valid(flame, preset_file)


def test_legacy_detector_is_strict(tmp_path):
    p = _write(tmp_path, "preset: darkquest\nlegacy_detector: {radius: 0.5}\n")
    with pytest.raises(ValueError, match="legacy_detector"):
        load_config(p)


def test_a_run_record_loads_back_as_the_recorded_config(tmp_path):
    from mcpsim.config import pin_supplied_acceptance, run_config
    from mcpsim.pipeline import write_run_record
    own = _load(tmp_path, {"beam": {"energy_gev": 122.0}, "detector": {"distance_m": 60.0},
                           "data": {"dy_cross": "my_scan.txt", "dy_ageo": "/work/my_ageo.txt",
                                    "acceptance": {"pi0": "/work/pi0.txt"}}})
    assert own.notes and not (own.geometry_is_preset or own.dy_cross_is_preset)
    cfgs = [own, pin_supplied_acceptance(own)]
    for name in ("darkquest", "ship", "lanl_12bar", "flame"):
        cfg = load_preset(name)
        cfgs += [cfg, run_config(replace(cfg, engine=replace(cfg.engine, fidelity="legacy")))]
    for i, cfg in enumerate(cfgs):
        out = tmp_path / f"run{i}"
        out.mkdir()
        assert load_config(write_run_record(cfg, out)) == cfg


def test_a_run_record_is_read_strictly(tmp_path):
    from mcpsim.pipeline import write_run_record
    path = write_run_record(load_preset("darkquest"), tmp_path)
    record = yaml.safe_load(path.read_text())
    for edited in ({**record, "beam": {"energy_gev": 400.0}},
                   {**record, "config": {**record["config"], "beam_energy": 400.0}}):
        path.write_text(yaml.safe_dump(edited, sort_keys=False))
        with pytest.raises(ValueError):
            load_config(path)


def test_config_round_trips_through_its_record():
    import json
    from dataclasses import asdict
    from mcpsim import load_preset
    from mcpsim.config import config_from_record
    for name in ("darkquest", "ship", "flame", "lanl_12bar"):
        cfg = load_preset(name)
        assert config_from_record(json.loads(json.dumps(asdict(cfg)))) == cfg
