import numpy as np
import pytest

from mcpsim.physics import sensitivity
from mcpsim.physics.brem import interp_to_grid
from mcpsim.physics.luminosity import effective_lumi_pb
from mcpsim.physics.meson_decay import I2, I3, meson_yield
from mcpsim.physics.constants import MESONS


def test_I2_unit_at_equal_args():
    # x == y collapses the ratio to 1.
    assert I2(0.0, 0.0) == pytest.approx(1.0)
    assert I2(0.1, 0.1) == pytest.approx(1.0)


def test_I2_decreases_towards_threshold():
    xs = [1e-4, 0.05, 0.15, 0.24999]
    vals = [I2(x, 1e-6) for x in xs]
    assert all(np.diff(vals) < 0)            # monotonically decreasing
    assert vals[-1] == pytest.approx(0.0, abs=2e-2)  # -> 0 at x -> 1/4


def test_I3_positive_decreasing_and_vanishes_at_threshold():
    assert I3(0.01) > I3(0.1) > I3(0.2) > 0
    assert I3(0.25) == pytest.approx(0.0, abs=1e-9)  # integral over empty range


def test_effective_lumi_darkquest_value():
    # DarkQuest: 1e20 POT, iron, one interaction length.
    L = effective_lumi_pb(1.0e20, 7.87, 16.8, 55.845)
    assert L == pytest.approx(1.4258e8, rel=1e-3)


def test_meson_yield_zero_above_threshold():
    spec = MESONS["pi0"]
    masses = np.array([0.01, spec.mass / 2 + 0.01, spec.mass])
    ageo = np.ones_like(masses)
    y = meson_yield(spec, masses, ageo, 1e20, 4.7)
    assert y[0] > 0
    assert y[1] == 0.0 and y[2] == 0.0


def test_detection_factor_monotonic_in_charge():
    eps = np.logspace(-5, 0, 50)
    d = sensitivity.detection_factor(eps, n_gamma=2.5e5, a=3)
    assert np.all(np.diff(d) > 0)
    assert np.all(d >= 0)


def test_brem_interp_zero_fills_outside_range():
    # Regression: brem must NOT constant-extrapolate beyond its mass range,
    # which previously inflated the high-mass yield (parity bug vs legacy).
    masses = np.array([0.1, 0.4])
    yields = np.array([10.0, 20.0])
    grid = np.array([0.01, 0.1, 0.2, 0.4, 4.0])
    out = interp_to_grid(masses, yields, grid)
    assert out[0] == 0.0                  # below range -> 0, not held constant
    assert out[-1] == 0.0                 # above range -> 0, not held constant
    assert out[1] == pytest.approx(10.0)  # at the low endpoint
    assert out[3] == pytest.approx(20.0)  # at the high endpoint
    assert 10.0 < out[2] < 20.0           # interpolated within range


def test_signal_2d_shape_and_scaling():
    baseline = np.array([1.0, 10.0, 100.0])
    eps = np.logspace(-3, 0, 7)
    sig = sensitivity.signal_2d(baseline, eps, n_gamma=2.5e5, a=3)
    assert sig.shape == (eps.size, baseline.size)
    # signal scales linearly with the baseline at fixed charge.
    assert sig[3, 2] == pytest.approx(100.0 * sig[3, 0])
