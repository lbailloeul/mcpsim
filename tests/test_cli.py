"""CLI behavior via CliRunner: clean errors, guards, exit codes."""

import pytest
from click.testing import CliRunner

from mcpsim.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_presets_lists_all(runner):
    r = runner.invoke(cli, ["presets"])
    assert r.exit_code == 0
    for name in ("darkquest", "ship", "lanl_12bar", "flame"):
        assert name in r.output


def test_unknown_preset_clean_error(runner):
    r = runner.invoke(cli, ["run", "--preset", "nope"])
    assert r.exit_code != 0
    assert "Unknown preset 'nope'" in r.output
    assert "Traceback" not in r.output


def test_missing_config_file(runner):
    r = runner.invoke(cli, ["run", "--config", "/no/such.yaml"])
    assert r.exit_code != 0
    assert "config file not found" in r.output


def test_mass_subset_requires_regenerate(runner):
    r = runner.invoke(cli, ["run", "--preset", "darkquest", "--mass-subset", "3"])
    assert r.exit_code != 0
    assert "--mass-subset only applies" in r.output


def test_no_submit_requires_condor(runner):
    r = runner.invoke(cli, ["generate", "brem", "--preset", "darkquest",
                            "--no-submit"])
    assert r.exit_code != 0
    assert "--no-submit only applies" in r.output


def test_missing_source_repo_actionable(runner, monkeypatch):
    monkeypatch.setenv("MCPSIM_SOURCE_REPO", "/definitely-nonexistent")
    r = runner.invoke(cli, ["run", "--preset", "darkquest"])
    assert r.exit_code != 0
    assert "MCPSIM_SOURCE_REPO" in r.output
    assert "Traceback" not in r.output


def test_config_typo_clean_error(runner, tmp_path):
    p = tmp_path / "typo.yaml"
    p.write_text("name: t\ndetector: {radius: 1.0}\n")
    r = runner.invoke(cli, ["run", "--config", str(p)])
    assert r.exit_code != 0
    assert "unknown key" in r.output
    assert "Traceback" not in r.output


def test_generate_dy_fails_loud_without_mg5(runner, mini_repo, tmp_path,
                                            monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # hide any mg5_aMC
    cfg = tmp_path / "mini.yaml"
    cfg.write_text(
        "name: minidy\nbeam: {energy_gev: 120.0}\n"
        "channels: [drell_yan]\n"
        "data: {mass_grid: mship_values.txt, dy_cross: dy_cross.txt, "
        "dy_ageo: dy_ageo.txt}\n"
    )
    r = runner.invoke(cli, ["generate", "dy", "--config", str(cfg),
                            "-o", str(tmp_path / "out")])
    assert r.exit_code != 0
    assert "produced no outputs" in r.output
