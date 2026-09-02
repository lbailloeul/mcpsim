"""Toolchain-free tests of the Python decay engine's numerics."""

import numpy as np
import pytest

from mcpsim.geometry import circle_hit_fraction, face_hit_fraction
from mcpsim.generation.pydecay import sample_costheta_rel, sample_dalitz_s


def test_face_kernel_matches_brute_force():
    rng = np.random.default_rng(11)
    n = 400_000
    w = h = 0.1
    for d, lo, hi in [(4.0, 3.85, 4.15), (0.0, 0.0, 0.3), (0.05, 0.0, 0.25)]:
        r = rng.uniform(lo, hi, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        x, y = r * np.cos(phi), r * np.sin(phi)
        brute = ((np.abs(x - d) <= w) & (np.abs(y) <= h)).mean()
        kernel = face_hit_fraction(r, d, w, h).mean()
        assert kernel == pytest.approx(brute, rel=0.05), f"offset {d}"


def test_face_kernel_containment_edges():
    # fully inside an on-axis face -> probability 1 (the pre-generalization
    # version returned 0.5 here)
    assert face_hit_fraction(np.array([0.03]), 0.0, 0.1, 0.1)[0] == 1.0
    assert face_hit_fraction(np.array([0.0]), 0.0, 0.1, 0.1)[0] == 1.0
    assert face_hit_fraction(np.array([0.0]), 5.0, 0.1, 0.1)[0] == 0.0
    # far outside the reach of the face
    assert face_hit_fraction(np.array([10.0]), 4.0, 0.1, 0.1)[0] == 0.0


def test_circle_kernel_on_axis_is_sharp_cut():
    r = np.array([0.1, 0.499, 0.5, 0.6])
    np.testing.assert_array_equal(circle_hit_fraction(r, 0.0, 0.5),
                                  [1.0, 1.0, 1.0, 0.0])


def test_circle_kernel_offaxis_matches_brute_force():
    rng = np.random.default_rng(5)
    n = 400_000
    r = rng.uniform(3.0, 5.0, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(phi), r * np.sin(phi)
    d, R = 4.0, 0.5
    brute = ((x - d) ** 2 + y ** 2 <= R * R).mean()
    kernel = circle_hit_fraction(r, d, R).mean()
    assert kernel == pytest.approx(brute, rel=0.03)


def test_dalitz_s_sampler_range_and_shape():
    rng = np.random.default_rng(3)
    mchi, mother = 0.02, 0.1349768
    s = sample_dalitz_s(rng, mchi, mother, 200_000)
    assert s.min() >= 4 * mchi ** 2
    assert s.max() <= mother ** 2
    # the marginal falls steeply with s: the median must sit near threshold
    assert np.median(s) < 0.25 * mother ** 2


def test_costheta_rel_distribution_moment():
    rng = np.random.default_rng(4)
    xi = np.full(400_000, 0.9)
    u = sample_costheta_rel(rng, xi)
    # E[u^2] for density ∝ (2-xi) + xi u^2 on [-1,1]:
    #   ((2-xi)/3 + xi/5) / ((2-xi) + xi/3)
    expected = ((2 - 0.9) / 3 + 0.9 / 5) / ((2 - 0.9) + 0.9 / 3)
    assert (u ** 2).mean() == pytest.approx(expected, rel=0.01)
    assert abs(u.mean()) < 0.01
