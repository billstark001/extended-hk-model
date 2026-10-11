import numpy as np
import pytest
from scipy.integrate import quad

from theory.release import gompertz_mean, gompertz_survival, mixture_mean, release_clock, stretched_mean
from theory.scaling import curve_distance


@pytest.mark.parametrize("hazard,acceleration", [(1.0, 0.0), (0.08, 0.4), (200.0, 0.001), (1.0, 0.01)])
def test_gompertz_mean_against_scaled_quadrature(hazard, acceleration):
    mean = gompertz_mean(hazard, acceleration)
    # Scale the integration variable so sharp early-time release is resolved.
    numerical = (
        quad(lambda s: float(gompertz_survival([s * mean], hazard, acceleration)[0]), 0, 100, epsabs=1e-11)[0] * mean
    )
    assert mean == pytest.approx(numerical, rel=1e-10)


def test_mean_retains_rebounds_and_long_tail_information():
    time = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    a = release_clock(time, [1.0, 0.4, 0.1, 0, 0], plateau=0)
    b = release_clock(time, [1.0, 0.4, 0.1, 0.3, 0], plateau=0)
    assert a.t50 == b.t50
    assert b.area > a.area
    assert b.rebound == pytest.approx(0.2)
    assert b.status == "known_plateau"


def test_plateau_status_and_unreleased_tail_are_explicit():
    time = np.linspace(0, 1, 100)
    clock = release_clock(time, np.exp(-time), plateau=0)
    assert clock.area == pytest.approx(1 - np.exp(-1), rel=1e-5)
    assert clock.survival[-1] > 0.3
    assert release_clock(time, np.exp(-time)).status == "endpoint_estimate"
    assert release_clock(time, np.ones_like(time)).status == "weak_release"
    with pytest.raises(ValueError, match="increase strictly"):
        release_clock([0, 0], [1, 0])


def test_stretched_and_mixture_means():
    assert stretched_mean(2.0, 1.0) == pytest.approx(0.5)
    assert stretched_mean(2.0, 2.0) == pytest.approx(np.sqrt(np.pi) / 4)
    assert mixture_mean(0.25, 1.0, 5.0) == 4.0


def test_scaling_uses_common_horizon_without_clamping():
    distance = curve_distance([0, 1, 2], [1, 0.5, 0], [0, 1], [1, 0.5])
    assert distance["relative_max"] == 0
    assert distance["horizon"] == 1
