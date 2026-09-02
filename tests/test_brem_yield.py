"""brem grid reading: cone cut, off-axis kernel consistency, mass parsing."""

import numpy as np
import pytest

from mcpsim.config import DetectorConfig
from mcpsim.physics.brem import brem_yield, brem_yield_offaxis


@pytest.fixture
def brem_dir(tmp_path):
    """Two synthetic mass files, flat sigma over a theta grid."""
    log10t = np.repeat(np.linspace(-4, -0.5, 10), 2)
    log10p = np.tile([0.0, 1.0], 10)
    sig = np.full((20, 3), 1.0)
    for m in ("0.01", "0.1"):
        np.savetxt(tmp_path / f"Brem_120GeV_{m}.txt",
                   np.column_stack([log10t, log10p, sig]))
    (tmp_path / "Brem_120GeV_badname_x.txt").write_text("0 0 1 1 1\n")
    return tmp_path


ARGS = dict(n_pot=1e20, density_g_cm3=7.87, interaction_length_cm=16.8, A=55.845)


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
