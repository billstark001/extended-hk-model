"""All-agent labels and unresolved probability mass; no major-mass filtering."""

import numpy as np

CATEGORIES = ("k1", "k2", "k3", "k4plus")


def category(k_all: int, absorbed: bool) -> str:
    if k_all < 1:
        raise ValueError("positive component count required")
    return CATEGORIES[min(k_all, 4) - 1] if absorbed else "unresolved"


def counts(labels):
    """Count every requested continuation; never renormalize away unresolved ones."""
    labels = list(labels)
    if not labels or any(x not in (*CATEGORIES, "unresolved") for x in labels):
        raise ValueError("nonempty physical labels required; task failures are not outcomes")
    return np.array([labels.count(k) for k in CATEGORIES], dtype=np.int64), labels.count("unresolved")


def allocation_bounds(observed, unresolved):
    """Bounds on R=1-sum(p**2) over all allocations of unresolved sample mass."""
    c = np.asarray(observed, dtype=float)
    if c.shape != (4,) or np.any(c < 0) or unresolved < 0 or c.sum() + unresolved <= 0:
        raise ValueError("four nonnegative counts and positive total required")
    total = c.sum() + unresolved
    concentrated = c.copy()
    concentrated[np.argmax(c)] += unresolved
    lo, hi = float(c.min()), float(c.max() + unresolved)
    for _ in range(80):
        level = (lo + hi) / 2
        if np.maximum(c, level).sum() > total:
            hi = level
        else:
            lo = level
    dispersed = np.maximum(c, (lo + hi) / 2)
    return max(0.0, 1 - float(np.sum((concentrated / total) ** 2))), max(
        0.0, 1 - float(np.sum((dispersed / total) ** 2))
    )
