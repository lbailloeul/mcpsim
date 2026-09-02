import numpy as np

from mcpsim import list_presets, load_preset
from mcpsim.config import _FloatLoader
import yaml


def test_scientific_notation_without_sign_parses_as_float():
    # The crux of an earlier bug: PyYAML SafeLoader reads '1.0e20' as a string.
    data = yaml.load("a: 1.0e20\nb: 2.5e5\nc: 1e-5\n", Loader=_FloatLoader)
    assert isinstance(data["a"], float) and data["a"] == 1e20
    assert isinstance(data["b"], float) and data["b"] == 2.5e5
    assert isinstance(data["c"], float) and data["c"] == 1e-5


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


def test_c_meson_rates_by_beam_energy():
    assert load_preset("darkquest").c_meson_rates()["pi0"] == 4.7
    assert load_preset("ship").c_meson_rates()["pi0"] == 7.5
    # low-energy preset only carries pi0/eta.
    rates = load_preset("lanl_12bar").c_meson_rates()
    assert set(rates) == {"pi0", "eta"}
