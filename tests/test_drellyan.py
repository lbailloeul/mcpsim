"""Drell-Yan normalization identity and pairing checks."""

import numpy as np
import pytest

from mcpsim import paths
from mcpsim.physics.constants import MB_TO_PB, SIGMA_PN_INELASTIC_MB
from mcpsim.physics.drellyan import dy_yield, row_mismatch


@pytest.fixture
def dy_files(tmp_path):
    masses = np.array([0.1, 0.5, 1.0, 2.0])
    sigma = np.array([3.0, 1.5, 0.8, 0.1])          # pb
    ageo = np.array([0.02, 0.015, 0.01, 0.005])
    cross_f = tmp_path / "cross.txt"
    ageo_f = tmp_path / "ageo.txt"
    np.savetxt(cross_f, sigma)
    np.savetxt(ageo_f, np.column_stack([masses, ageo]))
    return cross_f, ageo_f, masses, sigma, ageo


def test_normalization_identity_with_legacy(dy_files):
    cross_f, ageo_f, masses, sigma, ageo = dy_files
    n_pot = 1.0e20
    m, y = dy_yield(str(cross_f), str(ageo_f), n_pot)
    legacy = n_pot * (sigma * 1e-12) / 13e-3 * ageo
    np.testing.assert_allclose(y, legacy, rtol=1e-13)
    np.testing.assert_array_equal(m, masses)
    # the named constants reproduce the legacy magic number
    assert 1.0 / (SIGMA_PN_INELASTIC_MB * MB_TO_PB) == pytest.approx(
        1e-12 / 13e-3, rel=1e-14)


def test_row_mismatch_detection(tmp_path, dy_files):
    cross_f, ageo_f, *_ = dy_files
    assert row_mismatch(str(cross_f), str(ageo_f)) is None
    short = tmp_path / "short_cross.txt"
    np.savetxt(short, np.array([3.0, 1.5, 0.8]))
    msg = row_mismatch(str(short), str(ageo_f))
    assert msg and "3 rows" in msg and "4" in msg


def test_truncation_keeps_shorter_length(tmp_path, dy_files):
    _, ageo_f, *_ = dy_files
    short = tmp_path / "short_cross.txt"
    np.savetxt(short, np.array([3.0, 1.5]))
    m, y = dy_yield(str(short), str(ageo_f), 1e20)
    assert m.size == y.size == 2


def test_one_mass_scan(tmp_path):
    cross_f, ageo_f = tmp_path / "cross.txt", tmp_path / "ageo.txt"
    np.savetxt(cross_f, [2.6])                          # pb
    np.savetxt(ageo_f, [[0.3, 0.05]])
    m, y = dy_yield(str(cross_f), str(ageo_f), 1.0e20)
    assert m.shape == y.shape == (1,)
    assert m[0] == 0.3
    assert y[0] == pytest.approx(1.0e9, rel=1e-12)     # 1e20 POT x 2.6 pb / 13 mb x 0.05


def test_sigma_inel_per_nucleon_follows_target():
    from mcpsim.physics.constants import TARGET_MATERIALS as T
    from mcpsim.physics.drellyan import sigma_inel_per_nucleon_mb as s
    per = {k: s(T[k].density_g_cm3, T[k].interaction_length_cm) for k in ("iron", "molybdenum", "carbon")}
    assert per["iron"] == pytest.approx(12.56, rel=2e-3)        # ~ the published 13 mb
    assert per["molybdenum"] == pytest.approx(10.66, rel=2e-3)
    assert per["carbon"] == pytest.approx(19.35, rel=2e-3)


# -- mass-keyed per-material tables ------------------------------------------

def test_table_roundtrip(tmp_path):
    from mcpsim.physics.drellyan import load_table
    path = tmp_path / "dy_sigma_iron_120GeV.txt"
    path.write_text("# material: iron\n# acc_detector: 1 x 1 m face at 40 m\n"
                    "# columns: m_chi [GeV], sigma [pb], acc\n"
                    "0.01 100 0.04\n1.0 50 0.1\n4.0 1e-3 0.15\n")
    table = load_table(path)
    assert table.header["material"] == "iron"
    assert table.acc_detector == "1 x 1 m face at 40 m"
    np.testing.assert_array_equal(table.acc, [0.04, 0.1, 0.15])
    path.write_text("0.01 100\n1.0 50\n")                 # no acceptance column
    assert load_table(path).acc is None and load_table(path).acc_detector is None


def test_find_table_matches_energy_within_two_percent(tmp_path):
    from mcpsim.physics.drellyan import find_table
    for e in (120, 400):
        (tmp_path / f"dy_sigma_iron_{e}GeV.txt").write_text("1 1\n")
    assert find_table(tmp_path, 121.0, "iron").name == "dy_sigma_iron_120GeV.txt"
    assert find_table(tmp_path, 400.0, "iron").name == "dy_sigma_iron_400GeV.txt"
    assert find_table(tmp_path, 130.0, "iron") is None
    assert find_table(tmp_path, 120.0, "lead") is None


def test_yield_from_table_normalisation(tmp_path):
    from mcpsim.physics.drellyan import dy_yield_table, load_table
    path = tmp_path / "t.txt"
    path.write_text("0.1 30 0.01\n1.0 20 0.02\n")
    m, y = dy_yield_table(load_table(path), 1e20, 12.0)
    np.testing.assert_array_equal(m, [0.1, 1.0])
    np.testing.assert_allclose(y, 1e20 * np.array([30 * 0.01, 20 * 0.02]) / (12.0 * MB_TO_PB))
    path.write_text("0.1 30\n")
    with pytest.raises(ValueError, match="no acceptance column"):
        dy_yield_table(load_table(path), 1e20, 12.0)


def test_shipped_tables_cover_every_material_and_preset_energy():
    from mcpsim import geometry
    from mcpsim.config import load_preset
    from mcpsim.physics.constants import TARGET_MATERIALS
    from mcpsim.physics.drellyan import find_table, load_table
    faces = {120.0: geometry.face_signature(load_preset("darkquest").detector),
             400.0: geometry.face_signature(load_preset("ship").detector)}
    for energy, face in faces.items():
        sigma_1gev = {}
        for name, mat in TARGET_MATERIALS.items():
            table = load_table(find_table(paths.DY_TABLES_DIR, energy, name))
            m, s, h = table.masses, table.sigma_pb, table.header
            assert h["material"] == name and float(h["beam_energy_gev"]) == energy
            assert (int(h["A"]), int(h["Z"])) == (round(mat.A), mat.Z)
            assert np.all(np.diff(m) > 0) and np.all(s > 0)
            assert 2 * m[-1] < float(h["sqrt_s_gev"])
            # mass-independent below the 2 GeV pair-mass cut; vanishing at the end
            assert s[1] == pytest.approx(s[0], rel=1e-4) and s[-1] < 1e-9 * s[0]
            # the LO acceptance of the preset's 1 x 1 m face
            assert table.acc_detector == face
            assert np.all((table.acc > 0) & (table.acc < 1))
            sigma_1gev[name] = float(np.interp(1.0, m, s))
        # per nucleon: isospin and nuclear suppression order the materials
        assert (sigma_1gev["carbon"] > sigma_1gev["iron"] > sigma_1gev["molybdenum"]
                > sigma_1gev["tungsten"])


@pytest.mark.skipif(not paths.source_repo().exists(), reason="source repo not available")
@pytest.mark.parametrize("preset", ["darkquest", "ship"])
def test_iron_table_vs_the_paper_scan(preset):
    """Same material and settings as the paper's scan; only the PDFs differ
    (free-proton beam and the full iron nucleus vs the iron bound proton for
    both beams): +3-11% below 3 GeV, within -6% further up."""
    from mcpsim.config import load_preset
    from mcpsim.physics.drellyan import find_table, load_table
    cfg = load_preset(preset)
    scan = np.loadtxt(paths.resolve_data(cfg.data.dy_cross))
    mass = np.loadtxt(paths.resolve_data(cfg.data.dy_ageo))[:, 0]
    table = load_table(find_table(paths.DY_TABLES_DIR, cfg.beam.energy_gev, "iron"))
    m, s = table.masses, table.sigma_pb
    keep = mass <= (5.0 if preset == "darkquest" else 8.0)
    np.testing.assert_allclose(m[:keep.sum()], mass[keep], rtol=1e-6)   # same nodes
    ratio = s[:keep.sum()] / scan[keep]
    assert ratio.min() > 0.94 and ratio.max() < 1.12
    assert np.all(ratio[mass[keep] < 2.5] > 1.03)
