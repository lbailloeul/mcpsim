import numpy as np
import pytest

from mcpsim import load_preset
from mcpsim import geometry
from mcpsim.config import DetectorConfig


def test_cylindrical_theta_cut():
    assert geometry.cylindrical_theta_cut(0.5, 40.0) == np.arctan(0.5 / 40.0)
    # farther detector subtends a smaller angle.
    assert geometry.cylindrical_theta_cut(0.5, 60.0) < geometry.cylindrical_theta_cut(0.5, 40.0)


def test_bar_array_dims_and_equivalent_angle():
    det = DetectorConfig(type="bar_array", distance_m=6.0, bar_columns=3, bar_rows=2,
                         bar_layers=2, bar_size_m=0.05)
    w, h = geometry.bar_array_dims(det)
    assert w == pytest.approx(0.15) and h == pytest.approx(0.10)
    # equal-area disk radius -> arctan(r/L)
    r = np.sqrt(w * h / np.pi)
    assert geometry.equivalent_theta_cut(det) == np.arctan(r / det.distance_m)


def test_acceptance_theta_cut_dispatch():
    cyl = load_preset("darkquest").legacy_detector       # the paper's r = 0.5 m disc
    bar = load_preset("lanl_12bar").detector
    assert geometry.acceptance_theta_cut(cyl) == geometry.cylindrical_theta_cut(
        cyl.radius_m, cyl.distance_m)
    assert geometry.acceptance_theta_cut(bar) == geometry.equivalent_theta_cut(bar)


def test_face_signature_and_hit_fraction():
    face = load_preset("darkquest").detector
    assert geometry.face_signature(face) == "1 x 1 m face at 40 m"
    assert geometry.face_signature(load_preset("ship").legacy_detector) == "r = 0.5 m disc at 100 m"
    assert "4 mrad off axis" in geometry.face_signature(load_preset("flame").detector)
    r = np.array([0.0, 0.3, 0.6, 0.8])
    # on-axis square of half-width 0.5: inside, inside, corners only, outside
    np.testing.assert_allclose(geometry.hit_fraction(face, r)[[0, 1, 3]], [1, 1, 0])
    assert 0 < geometry.hit_fraction(face, r)[2] < 1
