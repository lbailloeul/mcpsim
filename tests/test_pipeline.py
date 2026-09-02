"""regenerate_inputs: warning propagation, guard ordering, early skips."""

from dataclasses import replace

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
