"""Independent invariants of the event-law closure and energy conventions."""

from itertools import product

import numpy as np
import pytest

from theory.energy import edge_energy, interaction_matrix, node_energy, potentials
from theory.exposure import selected_list_weights
from theory.recommendation import recommendation_kernel
from theory.rewiring import energy_rate, replacement_flux
from theory.state import PairState, opinion_grid
from theory.transport import synchronous_increment


def test_finite_counts_agree_with_enumerated_bernoulli_lists():
    # Enumeration is independent of the binomial count implementation.
    k, r, pn, pr = 2, 3, 0.25, 0.6
    expected = np.zeros(3)
    for bits in product((0, 1), repeat=k + r):
        probabilities = [pn] * k + [pr] * r
        mass = np.prod([p if b else 1 - p for b, p in zip(bits, probabilities)])
        m, n = sum(bits[:k]), sum(bits[k:])
        expected += mass * np.array([m / (m + n), n / (m + n), 0] if m + n else [0, 0, 1])
    actual = selected_list_weights(pn, pr, k, r)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert sum(actual) == pytest.approx(1)
    assert selected_list_weights(0, 0, k, r) == (0, 0, 1)


def test_replacement_matches_explicit_list_event_sum():
    rho = np.array([0.2, 0.5, 0.3])
    state = PairState.random_mixing(rho, 2)
    rec = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    mask = np.eye(3)
    result = replacement_flux(state, rec, mask, 2)
    exact = np.zeros((3, 3))
    for source in range(3):
        for neighbors in product(range(3), repeat=2):
            for candidates in product(range(3), repeat=2):
                mass = rho[source] * np.prod(rho[list(neighbors)]) * np.prod(rec[source, list(candidates)])
                bad = [j for j in neighbors if j != source]
                good = [j for j in candidates if j == source]
                if bad and good:
                    for j in bad:
                        exact[source, j] -= mass / len(bad)
                    for j in good:
                        exact[source, j] += mass / len(good)
    np.testing.assert_allclose(result.flux, exact, atol=1e-14)
    np.testing.assert_allclose(result.flux.sum(1), 0, atol=1e-14)


def test_energy_normalization_and_rewiring_nonpositivity():
    axis = opinion_grid(31)
    state = PairState.random_mixing(np.full(31, 1 / 31), 15)
    w = interaction_matrix(axis, 0.4)
    assert edge_energy(state.edge, w, 15) == pytest.approx(2 * node_energy(state.rho, w))
    node, edge = potentials(state, w)
    np.testing.assert_allclose(node, edge)
    mask = (abs(axis[:, None] - axis[None, :]) <= 0.4).astype(float)
    result = replacement_flux(state, recommendation_kernel(state, axis), mask, 10)
    assert energy_rate(result, w, 15, 0.2) <= 0
    np.testing.assert_allclose(result.flux.sum(1), 0, atol=1e-14)


def test_empty_sources_and_impossible_replacement():
    state = PairState.random_mixing([1.0, 0.0], 2)
    result = replacement_flux(state, [[1.0, 0.0], [1.0, 0.0]], np.eye(2), 3)
    np.testing.assert_array_equal(result.flux, np.zeros((2, 2)))
    assert np.isnan(potentials(state, interaction_matrix([-1.0, 1.0], 0.4))[1][1])
    with pytest.raises(ValueError, match="sum to one"):
        PairState.random_mixing([0.4, 0.4], 2)


def test_synchronous_decomposition_preserves_mixed_term():
    state = PairState.random_mixing([0.4, 0.6], 2)
    flux = np.array([[0.1, -0.1], [-0.2, 0.2]])
    m = np.array([[0.8, 0.2], [0.1, 0.9]])
    motion, rewire, mixed = synchronous_increment(state.edge, flux, m, 0.3)
    np.testing.assert_allclose(motion + rewire + mixed, m.T @ (state.edge + 0.3 * flux) @ m - state.edge, atol=1e-14)
    assert np.linalg.norm(mixed) > 0
    np.testing.assert_allclose((state.edge + motion + rewire + mixed).sum(1), state.degree * (state.rho @ m))
