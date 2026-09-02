"""Drell-Yan normalization identity and pairing checks."""

import numpy as np
import pytest

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
