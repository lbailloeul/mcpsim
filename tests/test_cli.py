"""CLI behavior via CliRunner: clean errors, guards, exit codes."""

import re

import numpy as np
import pytest
from click.testing import CliRunner

from mcpsim.cli import cli
from mcpsim.config import TARGET_ALIASES
from mcpsim.physics.constants import TARGET_MATERIALS


@pytest.fixture
def runner():
    return CliRunner()


def test_a_run_record_reruns_the_recorded_config(runner, mini_repo, tmp_path):
    import yaml
    src = tmp_path / "mini.yaml"
    # 122 GeV adds a config note and the detector override clears
    # geometry_is_preset, so the record carries the internal fields
    src.write_text("name: mini\nbeam: {energy_gev: 122.0}\nchannels: [mesons]\n"
                   "detector: {distance_m: 60.0}\n"
                   "data: {mass_grid: mship_values.txt, acceptance: {pi0: acc_pi0.txt}}\n")
    first = runner.invoke(cli, ["production", "-c", str(src), "-o", str(tmp_path / "a")])
    assert first.exit_code == 0, first.output
    again = runner.invoke(cli, ["production", "-c", str(tmp_path / "a" / "run_mini.yaml"),
                                "-o", str(tmp_path / "b")])
    assert again.exit_code == 0, again.output

    def record(d):
        return yaml.safe_load((tmp_path / d / "run_mini.yaml").read_text())

    assert record("a")["config"]["notes"] and not record("a")["config"]["geometry_is_preset"]
    assert record("b") == record("a")
    assert ((tmp_path / "b" / "production_mini.txt").read_text()
            == (tmp_path / "a" / "production_mini.txt").read_text())


def _production_column(path, column):
    names = path.read_text().splitlines()[0].lstrip("# ").split()
    return np.loadtxt(path)[:, names.index(column)]


def test_an_offaxis_move_recomputes_a_supplied_file_loudly(runner, mini_repo, tmp_path,
                                                           stub_recompute, flat_acceptance):
    own = flat_acceptance("pi0_own.txt", 0.4)
    pi0, output = {}, {}
    for name, data in (("own", f"data: {{acceptance: {{pi0: {own}}}}}\n"), ("preset", "")):
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(f"name: {name}\npreset: darkquest\nchannels: [mesons]\n{data}")
        r = runner.invoke(cli, ["production", "-c", str(cfg), "--offaxis", "4",
                                "-o", str(tmp_path / name)])
        assert r.exit_code == 0, r.output
        pi0[name] = _production_column(tmp_path / name / f"production_{name}_4mrad.txt", "pi0")
        output[name] = r.output
    np.testing.assert_array_equal(pi0["own"], pi0["preset"])
    assert str(own) in output["own"]


def test_a_file_the_config_names_survives_fidelity_legacy_on_the_cli(
        runner, mini_repo, tmp_path, stub_recompute, flat_acceptance):
    pi0 = {}
    for value in (0.4, 0.2):
        name = f"own{value * 10:.0f}"
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(f"name: {name}\npreset: darkquest\nchannels: [mesons]\n"
                       f"data: {{acceptance: {{pi0: {flat_acceptance(name + '.txt', value)}}}}}\n")
        r = runner.invoke(cli, ["production", "-c", str(cfg), "--fidelity", "legacy",
                                "-o", str(tmp_path / name)])
        assert r.exit_code == 0, r.output
        pi0[value] = _production_column(tmp_path / name / f"production_{name}_legacy.txt", "pi0")
    assert np.any(pi0[0.4] > 0)
    np.testing.assert_allclose(pi0[0.4], 2 * pi0[0.2], rtol=1e-7)


def test_condor_brem_bundles_ask_the_same_cores_from_both_commands(runner, mini_repo, tmp_path):
    def cpus(out, *cmd):
        r = runner.invoke(cli, [*cmd, "--preset", "darkquest", "--condor", "--no-submit",
                                "--mass-subset", "2", "-o", str(tmp_path / out)])
        assert r.exit_code == 0, r.output
        sub = (tmp_path / out / "generated" / "brem" / "brem.sub").read_text()
        return re.search(r"request_cpus\s*=\s*(\d+)", sub).group(1)

    assert cpus("gen", "generate", "brem") == cpus("run", "run", "--channels", "brem")
    assert cpus("gen6", "generate", "brem", "--n-proc", "6") == "6"


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


def test_preset_and_config_together_rejected(runner, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\n")
    r = runner.invoke(cli, ["run", "-p", "darkquest", "-c", str(p)])
    assert r.exit_code != 0
    assert "not both" in r.output


def test_channel_typo_rejected(runner, mini_repo):
    r = runner.invoke(cli, ["production", "-p", "darkquest", "--channels", "brems"])
    assert r.exit_code != 0
    assert "did you mean 'brem'" in r.output


def test_production_writes_its_plot_tables_and_run_record(runner, mini_repo, tmp_path):
    cfg = tmp_path / "mini.yaml"
    cfg.write_text("name: mini\nbeam: {energy_gev: 120.0}\nchannels: [mesons]\n"
                   "data: {mass_grid: mship_values.txt, acceptance: {pi0: acc_pi0.txt}}\n")
    out = tmp_path / "out"
    r = runner.invoke(cli, ["production", "-c", str(cfg), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "production_mini.pdf").exists()
    assert not (out / "limit_mini.pdf").exists()
    assert (out / "production_mini.txt").exists() and (out / "run_mini.yaml").exists()


def test_doctor_runs_without_data(runner, monkeypatch):
    monkeypatch.setenv("MCPSIM_SOURCE_REPO", "/definitely-nonexistent")
    r = runner.invoke(cli, ["doctor"])
    assert r.exit_code == 0
    assert "External toolchains" in r.output and "zenodo" in r.output


def test_target_swap_by_keyword(runner, mini_repo, tmp_path):
    cfg = tmp_path / "mini.yaml"
    cfg.write_text("name: mini\nbeam: {energy_gev: 120.0}\nchannels: [mesons]\n"
                   "data: {mass_grid: mship_values.txt, acceptance: {pi0: acc_pi0.txt}}\n")
    out = tmp_path / "out"
    r = runner.invoke(cli, ["production", "-c", str(cfg), "--target", "mo", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "molybdenum target" in r.output
    import yaml
    rec = yaml.safe_load((out / "run_mini_molybdenum.yaml").read_text())
    assert rec["config"]["target"]["name"] == "molybdenum"
    assert rec["config"]["target"]["A"] == 95.95


def test_target_typo_suggests(runner):
    r = runner.invoke(cli, ["run", "-p", "darkquest", "--target", "mobelium"])
    assert r.exit_code != 0
    assert "did you mean 'molybdenum'" in r.output


def test_run_help_names_every_target_material_and_alias(runner):
    r = runner.invoke(cli, ["run", "--help"])
    assert r.exit_code == 0, r.output
    target_help = r.output.split("--target", 1)[1].split("\n  -", 1)[0]
    words = set(re.findall(r"\w+", target_help))
    missing = [n for n in (*TARGET_MATERIALS, *TARGET_ALIASES) if n not in words]
    assert not missing, target_help


def test_compare_overlays_runs_and_writes_their_tables(runner, mini_repo, tmp_path):
    cfgs = []
    for name in ("mini a", "mini b"):
        c = tmp_path / f"{name.replace(' ', '_')}.yaml"
        c.write_text(f"name: {name}\nbeam: {{energy_gev: 120.0}}\nchannels: [mesons]\n"
                     "data: {mass_grid: mship_values.txt, acceptance: {pi0: acc_pi0.txt}}\n")
        cfgs += ["-c", str(c)]
    out = tmp_path / "cmp"
    r = runner.invoke(cli, ["compare", *cfgs, "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "limit_compare_mini_a_vs_mini_b.pdf").exists()
    assert (out / "limit_compare_mini_a_vs_mini_b.png").exists()
    for tag in ("mini_a", "mini_b"):
        assert (out / f"limit_curve_{tag}.txt").exists() and (out / f"run_{tag}.yaml").exists()
    assert runner.invoke(cli, ["compare"]).exit_code != 0       # nothing to compare


def test_legacy_runs_get_their_own_name_and_folder(runner, mini_repo, tmp_path):
    cfg = tmp_path / "mini.yaml"
    cfg.write_text("name: mini\nbeam: {energy_gev: 120.0}\nchannels: [mesons]\n"
                   "data: {mass_grid: mship_values.txt, acceptance: {pi0: acc_pi0.txt}}\n")
    r = runner.invoke(cli, ["limit", "-c", str(cfg), "--fidelity", "legacy", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "[mini legacy]" in r.output
    assert (tmp_path / "o" / "limit_mini_legacy.pdf").exists()
