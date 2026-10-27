"""Scientific contracts: unknown tails, recrossings, covariance and sampling units."""

import numpy as np
import pytest

from lifted.commitment import ensemble_cdf, timing
from lifted.committor import estimate, terminal_boundary, total_variation
from lifted.diagnostics import increment_factors
from lifted.outcomes import allocation_bounds, category


def test_unknown_tail_is_never_renormalized_or_a_physical_fifth_class():
    result = estimate(["k1"] * 3 + ["k2"] + ["unresolved"] * 4)
    assert sum(result["estimate"]) == 0.5
    assert result["unresolved"] == 4
    assert result["upper"][0] > 7 / 8
    with pytest.raises(ValueError, match="complete"):
        total_variation(result["estimate"], [1, 0, 0, 0])
    with pytest.raises(ValueError, match="physical"):
        estimate(["failed"])
    assert category(5, True) == "k4plus"


def test_probe_unanimity_and_simultaneous_family_do_not_certify_false_precision():
    one = estimate(["k1"] * 48)
    many = estimate(["k1"] * 48, family_size=100)
    assert one["lower"][0] < 0.95
    assert many["lower"][0] < one["lower"][0]
    assert estimate(["k1"] * 1024, family_size=100)["lower"][0] > 0.98


def test_allocation_bounds_cover_every_integer_assignment():
    low, high = allocation_bounds([2, 1, 0, 0], 3)
    for a in range(4):
        for b in range(4 - a):
            for c in range(4 - a - b):
                p = np.array([2 + a, 1 + b, c, 3 - a - b - c]) / 6
                assert low - 1e-14 <= 1 - np.sum(p * p) <= high + 1e-14


def test_first_hit_and_recrossing_are_separate_and_future_is_unobserved():
    rows = [
        dict(estimate=[p, 1 - p, 0, 0], lower=[lo, 0, 0, 0], upper=[hi, 0.2, 0.01, 0.01])
        for p, lo, hi in ((0.94, 0.91, 0.97), (0.85, 0.81, 0.89), (0.94, 0.91, 0.97))
    ]
    result = timing([1, 2, 3], rows)
    assert result["first_confirmed_tau"] == 1
    assert result["confirmed_returns_below"] == 1
    unknown = timing([1], [dict(estimate=[0.6, 0.4, 0, 0], lower=[0.5, 0.3, 0, 0], upper=[0.7, 0.5, 0.1, 0.1])])
    cdf = ensemble_cdf([dict(initial=0, **unknown)], [1, 2])
    assert cdf[0]["cdf_upper"] == 0 and cdf[1]["cdf_upper"] == 1


def test_covariance_factors_preserve_cross_block_and_fixed_degree_constraint():
    node = np.array([[0.1, -0.1], [-0.1, 0.1], [0.2, -0.2]])
    edge = np.zeros((3, 2, 2))
    edge[:, :, 0] = 3 * node
    x = np.column_stack([node, edge.reshape(3, -1)])
    mean, factor = increment_factors(x, bins=2, degree=3)
    np.testing.assert_allclose(mean, x.mean(axis=0))
    np.testing.assert_allclose(factor.T @ factor, np.cov(x, rowvar=False))
    assert np.any((factor.T @ factor)[:2, 2:])
    with pytest.raises(ValueError, match="constraint"):
        increment_factors(x + 1, bins=2, degree=3)


def test_ensemble_weights_initial_states_equally():
    rows = [dict(initial=0, first_possible_tau=0, first_confirmed_tau=0, observed_through_tau=1)]
    rows.extend(dict(initial=1, first_possible_tau=2, first_confirmed_tau=2, observed_through_tau=2) for _ in range(10))
    assert ensemble_cdf(rows, [0])[0]["cdf_lower"] == 0.5


def test_pending_work_is_unknown_probability_mass_but_not_physical_censoring():
    result = estimate(["k1"] * 8, pending=120)
    assert result["paths"] == 128 and result["pending_paths"] == 120 and result["unresolved"] == 0
    assert result["estimate"][0] == 8 / 128
    assert result["upper"][1] > 0.9
    empty = estimate([], pending=128)
    assert empty["lower"] == [0] * 4 and empty["upper"] == [1] * 4


def test_absorbing_boundary_is_exact_even_without_probe_samples():
    p = terminal_boundary(estimate([], pending=128), "k2")
    assert p["estimate"] == p["lower"] == p["upper"] == [0, 1, 0, 0]
    assert timing([16], [p], delta=0.05)["first_confirmed_tau"] == 16
    with pytest.raises(ValueError, match="contradict"):
        terminal_boundary(estimate(["k1"]), "k2")
