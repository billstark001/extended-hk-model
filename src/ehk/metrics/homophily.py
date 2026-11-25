"""Homophily-index normalization shared by microscopic and kinetic analyses."""


def uniform_concordance_probability(epsilon: float) -> float:
    """Return P(|X-Y| <= epsilon) for iid X,Y ~ Uniform[-1, 1]."""

    if not 0 <= epsilon <= 2:
        raise ValueError("epsilon must be in [0, 2]")
    return epsilon - epsilon**2 / 4


def normalize_homophily(raw_homophily: float, epsilon: float) -> float:
    """Normalize random mixing to zero and complete concordance to one."""

    baseline = uniform_concordance_probability(epsilon)
    if baseline >= 1:
        return 1.0 if raw_homophily >= 1 else 0.0
    return max(0.0, (raw_homophily - baseline) / (1 - baseline))


def mean_followee_edge_concordance(graph, opinions, epsilon: float) -> float:
    """Average each positive-out-degree user's concordant follow-edge fraction."""
    import numpy as np

    edges = np.asarray(list(graph.edges), dtype=int).reshape(-1, 2)
    if not len(edges):
        return float("nan")
    x = np.asarray(opinions, dtype=float)
    sources, targets = edges.T
    concordant = np.abs(x[sources] - x[targets]) < epsilon
    degree = np.bincount(sources)
    close = np.bincount(sources, weights=concordant)
    valid = degree > 0
    return float(np.mean(close[valid] / degree[valid]))
