"""Conditional outcome estimates and fixed-family simultaneous sampling bounds."""

import numpy as np
from scipy.stats import beta  # type: ignore[import-untyped]

from lifted.outcomes import CATEGORIES, allocation_bounds, counts
from lifted.types import Probability


def estimate(labels, *, confidence=0.95, family_size=1, pending=0) -> Probability:
    """Bonferroni/Clopper-Pearson bounds include all possible unresolved labels.

    family_size counts the predeclared queried states, each with four classes.
    Selection and refinement use independent samples. Estimates retain unknown
    mass and consequently sum to less than one when continuations are unresolved.
    """
    if not 0 < confidence < 1 or family_size < 1 or pending < 0:
        raise ValueError("valid confidence and positive family_size required")
    labels = list(labels)
    c, u = counts(labels) if labels else (np.zeros(4, dtype=np.int64), 0)
    n = int(c.sum() + u + pending)
    if n == 0:
        raise ValueError("nonempty planned sampling family required")
    unknown = u + pending
    tail = (1 - confidence) / (8 * family_size)
    lower = [0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1)) for k in c]
    upper = [1.0 if k + unknown == n else float(beta.ppf(1 - tail, k + unknown + 1, n - k - unknown)) for k in c]
    rlo, rhi = allocation_bounds(c, unknown)
    return dict(
        paths=n,
        counts=c.tolist(),
        unresolved=u,
        pending_paths=pending,
        estimate=(c / n).tolist(),
        lower=lower,
        upper=upper,
        disagreement_lower=rlo,
        disagreement_upper=rhi,
    )


def total_variation(left, right):
    """TV between complete four-class probability vectors; unresolved vectors rejected."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError("four finite nonnegative probabilities required")
    if not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1):
        raise ValueError("TV requires complete distributions")
    return float(np.abs(a - b).sum() / 2)


def terminal_boundary(probability: Probability, label: str) -> Probability:
    """Impose the exact absorbing-label boundary, independently of probe budget.

    The caller supplies a classification of the queried state, not unanimous
    future samples. Observed labels must agree with that boundary.
    """
    index = CATEGORIES.index(label)
    if probability["unresolved"] or any(n and i != index for i, n in enumerate(probability["counts"])):
        raise ValueError("continuations contradict the terminal boundary")
    exact = [float(i == index) for i in range(4)]
    return {
        **probability,
        "estimate": exact,
        "lower": exact.copy(),
        "upper": exact.copy(),
        "disagreement_lower": 0.0,
        "disagreement_upper": 0.0,
    }
