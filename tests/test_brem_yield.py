"""brem grid reading: cone cut, off-axis kernel consistency, mass parsing,
target rescaling."""

import numpy as np
import pytest

from mcpsim.config import DetectorConfig
from mcpsim.physics.brem import (brem_yield, brem_yield_offaxis, grid_target_A,
                                 sigma_pA_mb)
from mcpsim.physics.constants import TARGET_MATERIALS


def _write_grids(directory, header=""):
    log10t = np.repeat(np.linspace(-4, -0.5, 10), 2)
    log10p = np.tile([0.0, 1.0], 10)
    sig = np.full((20, 3), 1.0)
    for m in ("0.01", "0.1"):
        np.savetxt(directory / f"Brem_120GeV_{m}.txt",
                   np.column_stack([log10t, log10p, sig]), header=header)


@pytest.fixture
def brem_dir(tmp_path):
    """Two synthetic mass files, flat sigma over a theta grid, no header."""
    _write_grids(tmp_path)
    (tmp_path / "Brem_120GeV_badname_x.txt").write_text("0 0 1 1 1\n")
    return tmp_path


@pytest.fixture
def iron_brem_dir(tmp_path):
    """The same grids with the archived DarkQuest header declaring iron."""
    d = tmp_path / "iron"
    d.mkdir()
    _write_grids(d, header="E_beam = 120 GeV   # beam energy\n"
                           "A_Fe = 55.845 g/mol   # atomic mass of iron")
    return d


ARGS = dict(n_pot=1e20, density_g_cm3=7.87, interaction_length_cm=16.8, A=55.845)


def _target_args(name):
    t = TARGET_MATERIALS[name]
    return dict(n_pot=1e20, density_g_cm3=t.density_g_cm3,
                interaction_length_cm=t.interaction_length_cm, A=t.A)


def test_grid_target_read_from_header(brem_dir, iron_brem_dir):
    assert grid_target_A(str(iron_brem_dir / "Brem_120GeV_0.01.txt")) == pytest.approx(55.845)
    assert grid_target_A(str(brem_dir / "Brem_120GeV_0.01.txt")) is None


def test_matching_target_is_not_rescaled(brem_dir, iron_brem_dir):
    _, y_hdr = brem_yield(str(iron_brem_dir), "Brem_120GeV_*.txt", 0.01, **ARGS)
    _, y_raw = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", 0.01, **ARGS)
    np.testing.assert_allclose(y_hdr, y_raw, rtol=1e-12)


def test_other_target_rescales_sigma_pA(brem_dir, iron_brem_dir):
    carbon = _target_args("carbon")
    ratio = sigma_pA_mb(carbon["A"]) / sigma_pA_mb(55.845)
    _, y_hdr = brem_yield(str(iron_brem_dir), "Brem_120GeV_*.txt", 0.01, **carbon)
    _, y_raw = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", 0.01, **carbon)
    np.testing.assert_allclose(y_hdr, y_raw * ratio, rtol=1e-12)
    det = DetectorConfig(type="bar_array", distance_m=1000.0, bar_columns=4,
                         bar_rows=4, bar_size_m=0.05, offaxis_mrad=4.0)
    _, o_hdr = brem_yield_offaxis(str(iron_brem_dir), "Brem_120GeV_*.txt", det, **carbon)
    _, o_raw = brem_yield_offaxis(str(brem_dir), "Brem_120GeV_*.txt", det, **carbon)
    np.testing.assert_allclose(o_hdr, o_raw * ratio, rtol=1e-12)


def test_consistent_targets_give_same_per_pot_brem(iron_brem_dir):
    """A thick target gives ~one interaction per POT whatever the material, so
    graphite and iron read from the same iron grids must agree to a few %.
    Swapping only the luminosity (the old behaviour) made graphite x3."""
    _, y_fe = brem_yield(str(iron_brem_dir), "Brem_120GeV_*.txt", 0.01, **_target_args("iron"))
    _, y_c = brem_yield(str(iron_brem_dir), "Brem_120GeV_*.txt", 0.01, **_target_args("carbon"))
    np.testing.assert_allclose(y_c / y_fe, 1.0, atol=0.02)


def test_masses_parsed_and_sorted(brem_dir):
    m, y = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", 0.01, **ARGS)
    # the unparsable filename is skipped, the rest sorted by mass
    np.testing.assert_array_equal(m, [0.01, 0.1])
    assert np.all(y >= 0)


def test_cone_cut_monotone_in_angle(brem_dir):
    _, y_small = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", 1e-3, **ARGS)
    _, y_large = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", 0.3, **ARGS)
    assert np.all(y_large >= y_small)
    assert np.any(y_large > y_small)


def test_offaxis_kernel_huge_face_captures_everything(brem_dir):
    """A face far larger than the whole theta range must recover the full
    integral (every bin weight -> 1)."""
    det = DetectorConfig(type="bar_array", distance_m=100.0, bar_columns=200,
                         bar_rows=200, bar_size_m=1.0, offaxis_mrad=0.0)
    _, y_face = brem_yield_offaxis(str(brem_dir), "Brem_120GeV_*.txt", det, **ARGS)
    _, y_all = brem_yield(str(brem_dir), "Brem_120GeV_*.txt", np.pi / 2, **ARGS)
    np.testing.assert_allclose(y_face, y_all, rtol=1e-6)


def test_offaxis_yield_decreases_with_offset(brem_dir):
    common = dict(type="bar_array", distance_m=100.0, bar_columns=4,
                  bar_rows=4, bar_size_m=0.05)
    y = []
    for angle in (0.0, 30.0):
        det = DetectorConfig(offaxis_mrad=angle, **common)
        _, yi = brem_yield_offaxis(str(brem_dir), "Brem_120GeV_*.txt", det, **ARGS)
        y.append(yi)
    assert np.all(y[0] >= y[1])


def test_face_max_theta():
    from mcpsim.geometry import face_max_theta
    flame = DetectorConfig(type="bar_array", distance_m=1000.0, bar_columns=4,
                           bar_rows=4, bar_size_m=0.05, offaxis_mrad=14.0)
    d = 1000.0 * np.tan(14e-3)
    assert face_max_theta(flame) == pytest.approx(np.arctan(np.hypot(d + 0.1, 0.1) / 1000.0))
    ship = DetectorConfig(type="cylindrical", distance_m=100.0, radius_m=0.5)
    assert 1e3 * face_max_theta(ship) == pytest.approx(5.0, rel=1e-4)


def _brem_warnings(cfg, grids):
    from dataclasses import replace
    from mcpsim import model
    cfg = replace(cfg, data=replace(cfg.data, brem_dir=str(grids)))
    grid = np.array([0.01, 0.1])
    res = model.ProductionResult(config=cfg, masses=grid)
    model._add_brem(cfg, grid, res)
    return [w for w in res.warnings if "not accurate above" in w]


def test_brem_validity_angle_warning(iron_brem_dir):
    """Grids resolve the generating face only out to its half-width: FLAME
    past 12.5 mrad is warned (collapse), and so are the corners of the
    DarkQuest/SHiP 1 x 1 m face; FLAME at 4 mrad and the paper's discs not."""
    from dataclasses import replace
    from mcpsim import load_preset
    flame = load_preset("flame")
    far = replace(flame, detector=replace(flame.detector, offaxis_mrad=14.0))
    assert len(_brem_warnings(far, iron_brem_dir)) == 1
    assert "12.5 mrad" in _brem_warnings(far, iron_brem_dir)[0]
    assert "collapses at low mass" in _brem_warnings(far, iron_brem_dir)[0]
    assert _brem_warnings(flame, iron_brem_dir) == []          # 4 mrad default
    for name in ("darkquest", "ship"):
        cfg = load_preset(name)
        w = _brem_warnings(cfg, iron_brem_dir)                  # the face's corners
        assert len(w) == 1 and "face beyond that angle is missing" in w[0]
        assert _brem_warnings(replace(cfg, detector=cfg.legacy_detector), iron_brem_dir) == []


def test_zero_brem_warning_tells_no_grid_from_zero_yield(mini_repo, mini_config):
    from dataclasses import replace
    from mcpsim.model import compute_production
    (mini_repo / "empty-brem").mkdir()
    zero = mini_repo / "zero-brem"
    zero.mkdir()
    np.savetxt(zero / "Brem_120GeV_0.02.txt",
               np.column_stack([[-4.0, -3.0], [0.0, 1.0], np.zeros((2, 3))]))

    def zero_brem_warnings(brem_dir):
        cfg = replace(mini_config, channels=["brem"],
                      data=replace(mini_config.data, brem_dir=brem_dir))
        result = compute_production(cfg)
        assert not result.brem_central.any()
        return [w for w in result.warnings if cfg.data.brem_pattern in w]

    [no_grid] = zero_brem_warnings("empty-brem")
    [zero_yield] = zero_brem_warnings("zero-brem")
    assert "no brem files matched" in no_grid
    assert "no brem files matched" not in zero_yield
