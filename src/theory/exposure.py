"""Finite-list probabilities under conditionally iid exposure closure."""

import numpy as np
from scipy.stats import binom


def at_least_one(probability, count: int):
    """Return 1-(1-p)**count stably for probabilities in [0,1]; zero opportunities give zero probability."""
    probability = np.asarray(probability, dtype=float)
    if count < 0 or int(count) != count or not np.isfinite(probability).all():
        raise ValueError("finite probabilities and a nonnegative integer count required")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    if count == 0:
        return np.zeros_like(probability)
    with np.errstate(divide="ignore"):
        return -np.expm1(count * np.log1p(-probability))


def concordant_counts(
    followee_probability: float, recommendation_probability: float, degree: int, recommendations: int
) -> np.ndarray:
    """Joint law of the two concordant counts, including the empty list."""
    at_least_one(followee_probability, degree)
    at_least_one(recommendation_probability, recommendations)
    return np.outer(
        binom.pmf(np.arange(degree + 1), degree, followee_probability),
        binom.pmf(np.arange(recommendations + 1), recommendations, recommendation_probability),
    )


def selected_list_weights(
    followee_probability: float, recommendation_probability: float, degree: int, recommendations: int
) -> tuple[float, float, float]:
    """Expected HK fractions / Deffuant selection weights; no mean denominator.

    The first two weights sum to the update probability. The third is the
    empty-list no-update probability. For HK, these determine the mean only;
    they do not reconstruct the distribution of random list averages.
    """
    law = concordant_counts(followee_probability, recommendation_probability, degree, recommendations)
    m, n = np.indices(law.shape)
    total = m + n
    follow = np.divide(m, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    rec = np.divide(n, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    return float(np.sum(law * follow)), float(np.sum(law * rec)), float(law[0, 0])
