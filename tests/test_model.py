"""compute_production assembly, warnings, and geometry honesty (mini repo)."""

from dataclasses import replace

import numpy as np
import pytest

from mcpsim.model import _load_acceptance_on_grid, compute_production


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
    cfg = replace(mini_config, family="flame")
    result = compute_production(cfg)
    assert "upsilon" not in result.meson_channels
    assert any("upsilon" in w for w in result.warnings)


def test_upsilon_data_override(mini_config):
    cfg = replace(mini_config, family="flame",
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


def test_short_single_column_acceptance_raises(tmp_path):
    f = tmp_path / "one_col.txt"
    np.savetxt(f, np.array([1e-3, 2e-3, 3e-3]))   # 3 rows, 1 col; grid has 5
    with pytest.raises(ValueError, match="single-column acceptance"):
        _load_acceptance_on_grid(f, np.linspace(0.01, 0.1, 5))


def test_config_notes_reach_warnings(mini_config):
    cfg = replace(mini_config, notes=["synthetic note"])
    result = compute_production(cfg)
    assert "synthetic note" in result.warnings
