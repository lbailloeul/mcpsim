"""Tests for the parts of the generation layer that need no heavy toolchain:
the LHE acceptance parser and the bremsstrahlung global-patcher (text only)."""

import numpy as np
import pytest

from mcpsim import load_preset, paths
from mcpsim.generation import brem_run, mesons
from mcpsim.generation.madgraph import MCP_PDG_DEFAULT, parse_lhe_acceptance

_SOURCE_REPO_PRESENT = paths.source_repo().exists()

# Two-event LHE: a forward pair (both chi on the face), a wide-angle pair (none).
_LHE = f"""\
<LesHouchesEvents version="3.0">
<init>
</init>
<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 6.0e+01 6.0e+01 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+01 1.0e+01 0 0 -1
 {MCP_PDG_DEFAULT} 1 1 2 0 0 0.0 0.0 5.0e+01 5.0e+01 1.0e-01 0 1
-{MCP_PDG_DEFAULT} 1 1 2 0 0 0.0 0.0 1.0e+00 1.0e+00 1.0e-01 0 -1
</event>
<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 6.0e+01 6.0e+01 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+01 1.0e+01 0 0 -1
 {MCP_PDG_DEFAULT} 1 1 2 0 0 5.0e+01 0.0 1.0e+00 5.0e+01 1.0e-01 0 1
-{MCP_PDG_DEFAULT} 1 1 2 0 0 -5.0e+01 0.0 1.0e+00 5.0e+01 1.0e-01 0 -1
</event>
</LesHouchesEvents>
"""


def test_parse_lhe_acceptance(tmp_path):
    lhe = tmp_path / "events.lhe"
    lhe.write_text(_LHE)
    cfg = load_preset("darkquest")  # 1 x 1 m face at 40 m
    n_total, n_hits = parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg)
    assert n_total == 2
    assert n_hits == 2  # both chi of the forward pair; per-chi acceptance 2 / (2 * 2)


def test_lhe_events_are_boosted_to_the_lab(tmp_path):
    """mcpsim's fixed-target runs write centre-of-mass events: a chi at
    100 mrad there hits the face only once boosted by y_cm = 2.77."""
    from mcpsim.generation.madgraph import lab_rapidity
    cfg = load_preset("darkquest")
    assert lab_rapidity(cfg) == pytest.approx(2.7722, abs=1e-4)
    lhe = tmp_path / "cm.lhe"
    lhe.write_text(f"""<event>
 4 1 +1.0e-03 1.0e02 7.5e-03 1.2e-01
 2 -1 0 0 501 0 0 0 1.0e+00 1.0e+00 0 0 1
-2 -1 0 0 0 501 0 0 -1.0e+00 1.0e+00 0 0 -1
 {MCP_PDG_DEFAULT} 1 1 2 0 0 0.0995 0.0 0.995 1.0 1.0e-03 0 1
-{MCP_PDG_DEFAULT} 1 1 2 0 0 -0.0995 0.0 -0.995 1.0 1.0e-03 0 -1
</event>
""")
    assert parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg) == (1, 0)
    # lab angle 0.0995 / (0.995 cosh y + sinh y) = 6.2 mrad: on the face
    assert parse_lhe_acceptance(lhe, MCP_PDG_DEFAULT, cfg, boost_y=lab_rapidity(cfg)) == (1, 1)


@pytest.mark.skipif(not _SOURCE_REPO_PRESENT, reason="source repo (mCP_brem.py) unavailable")
def test_brem_patcher_rewrites_globals(tmp_path):
    cfg = load_preset("ship")  # 400 GeV, molybdenum -> different globals than the script default
    patched = brem_run._prepare_patched_script(cfg, tmp_path)
    text = patched.read_text()
    # Patched copy lives in the work dir, not the source repo.
    assert patched.parent == tmp_path
    assert f"{cfg.beam.n_pot:.6e}" in text          # N_POT updated (2.000000e+20)
    assert "95.95" in text                           # molybdenum A
    # sqrt(s) recomputed for 400 GeV, no longer the 15.06 GeV default.
    assert "15.06)**2" not in text and "15.06**2" not in text
    # The output header names the nucleus sigma_pA used, so the readers do not
    # rescale the regenerated grids by the hard-coded iron A.
    assert '"A_target = 95.95 g/mol' in text
    assert '"A_Fe = 55.845 g/mol' not in text
    # the generator's square face, at SHiP's distance, and files named for 400 GeV
    assert "z_det    = 100.0" in text and "n_bar    = 20" in text and "bar_size = 0.05" in text
    assert 'f"Brem_400GeV_{m_chi:g}.txt"' in text and "Brem_120GeV_" not in text


def test_condor_bundle(tmp_path):
    cfg = load_preset("darkquest")
    out = brem_run.prepare_condor_brem(cfg, np.array([0.001, 0.05, 0.2]), tmp_path,
                                       n_proc=8, submit=False)
    assert out == tmp_path / "brem_out" and out.is_dir()
    wrapper = (tmp_path / "run_brem_job.sh").read_text()
    # absolute bundle path (a worker runs its own scratch copy of the script),
    # set -u off around the LCG setup, the canonical eVMD mode, a venv python
    assert f'WORKDIR="${{BREM_WORKDIR:-{tmp_path.resolve()}}}"' in wrapper
    assert "set +u" in wrapper and "--form-factor-mode full" in wrapper
    assert "brem_venv/bin/python" in wrapper and "Missing Python modules" in wrapper
    assert "queue 3" in (tmp_path / "brem.sub").read_text()
    assert "condor_submit brem.sub" in (tmp_path / "README_condor.txt").read_text()
    np.testing.assert_allclose(np.loadtxt(tmp_path / "brem_masses.txt"), [0.001, 0.05, 0.2])


def test_generator_face():
    from dataclasses import replace
    face = load_preset("darkquest").detector
    assert brem_run._generator_face(face) == (20, 0.05)
    assert brem_run._generator_face(load_preset("darkquest").legacy_detector) == (20, 0.05)
    with pytest.raises(brem_run.common.ToolchainError, match="off-axis"):
        brem_run._generator_face(load_preset("flame").detector)
    with pytest.raises(brem_run.common.ToolchainError, match="square"):
        brem_run._generator_face(replace(face, bar_rows=10))


def test_meson_backend_dispatch():
    assert mesons.choose_backend(load_preset("darkquest")) == "pythia"
    assert mesons.choose_backend(load_preset("ship")) == "pythia"
    assert mesons.choose_backend(load_preset("lanl_12bar")) == "burman_smith"


def test_bundled_brem_generator_is_valid_python():
    # a C-style '// Author' line once broke every brem regeneration
    import py_compile
    py_compile.compile(str(paths.mcp_brem_script()), doraise=True)


def test_dy_condor_bundle(tmp_path):
    import json
    import py_compile
    import subprocess
    from mcpsim.config import config_from_record
    from mcpsim.generation import madgraph
    cfg = load_preset("darkquest")
    dy_cross, dy_ageo = madgraph.prepare_condor_dy(cfg, np.array([0.001, 0.5, 2.0]), tmp_path,
                                                   n_proc=4, submit=False)
    bundle = tmp_path.resolve()
    # the files the batch fills once its last job is done
    assert (dy_cross, dy_ageo) == (bundle / "dy_cross_darkquest.txt", bundle / "ageo_dy_darkquest.txt")
    assert not dy_cross.exists()
    wrapper = tmp_path / "run_dy_job.sh"
    text = wrapper.read_text()
    # absolute bundle path (a worker runs its own scratch copy), set -u off
    # around the LCG setup, a venv python, the bundle's PDF copy first
    assert f'WORKDIR="${{DY_WORKDIR:-{bundle}}}"' in text and "set +u" in text
    assert "dy_venv/bin/python" in text and "lhapdf/pdfsets.index" in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    sub = (tmp_path / "dy.sub").read_text()
    assert "queue 3" in sub and "request_cpus   = 4" in sub
    # each job rebuilds the run's config, and has the model and masses at hand
    assert config_from_record(json.loads((tmp_path / "config.json").read_text())) == cfg
    assert json.loads((tmp_path / "bundle.json").read_text())["model"] == "mcp_ufo"
    assert (tmp_path / "mcp_ufo" / "particles.py").exists()
    np.testing.assert_allclose(np.loadtxt(tmp_path / "dy_masses.txt"), [0.001, 0.5, 2.0])
    py_compile.compile(str(tmp_path / "dy_job.py"), doraise=True)
    readme = (tmp_path / "README_condor.txt").read_text()
    assert "condor_submit dy.sub" in readme and str(dy_cross) in readme
    assert "AlphaS_NumFlavors" in readme and "dy_material: iron" in readme


def test_dy_condor_jobs_assemble_the_scan(tmp_path, monkeypatch, capsys):
    """Jobs run (MadGraph replaced by a stub writing _LHE) in the scratch dir,
    write one line each, and the last one assembles the scan."""
    from pathlib import Path

    from mcpsim.generation import madgraph
    from mcpsim.generation.common import ToolchainError

    def fake_mg(cfg, model, mchi, proc_dir, *, nevents, cores=None):
        assert Path(model).parent == proc_dir.parent        # the job's own model copy
        run = proc_dir / "Events" / "run_01"
        run.mkdir(parents=True)
        (run / "run_01_tag_1_banner.txt").write_text(f"<MGVersion> {nevents} {cores}\n")
        (run / "unweighted_events.lhe").write_text(_LHE)
        return 100.0 * (1 + mchi), run / "unweighted_events.lhe"

    monkeypatch.setattr(madgraph, "_run_single_mass", fake_mg)
    bundle = tmp_path / "dy"
    madgraph.prepare_condor_dy(load_preset("darkquest"), np.array([0.001, 0.5, 2.0]), bundle,
                               n_proc=2, submit=False)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("_CONDOR_SCRATCH_DIR", str(scratch))
    madgraph.main(["job", str(bundle), "0"])
    assert (scratch / "dy_job_0000" / "mcp_ufo").is_dir() and not (bundle / "work").exists()
    with pytest.raises(ToolchainError, match=r"2 of 3 .* \(index 1, 2\)"):
        madgraph.collect_condor_dy(bundle)
    madgraph.main(["job", str(bundle), "2", "--nevents", "1000"])
    madgraph.main(["job", str(bundle), "1"])                  # the last one assembles
    assert "different number of events" in capsys.readouterr().out   # job 2
    np.testing.assert_allclose(np.loadtxt(bundle / "dy_cross_darkquest.txt"), [100.1, 150.0, 300.0])
    # both chi of the forward pair hit: per-chi acceptance 2 / (2 * 2)
    np.testing.assert_allclose(np.loadtxt(bundle / "ageo_dy_darkquest.txt"),
                               [[0.001, 0.5], [0.5, 0.5], [2.0, 0.5]])
    assert (bundle / "dy_out" / "banner_0002.txt").read_text() == "<MGVersion> 1000 2\n"
    with pytest.raises(SystemExit, match="out of range"):
        madgraph.main(["job", str(bundle), "3"])
