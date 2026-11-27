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
    from .state import MeasurementState

    return mean_state_edge_concordance(MeasurementState.from_graph(graph, opinions), epsilon)


def mean_state_edge_concordance(state, epsilon: float) -> float:
    """Reuse a measurement state's edge distances and positive out-degrees.

    Historical float32 opinions are promoted before subtracting, as in the
    original homophily calculation. Float64 state measurements reuse the same
    edge-distance array as the distance indices.
    """
    import numpy as np

    edges = state.edges
    if not len(edges):
        return float("nan")
    x = np.asarray(state.opinions, dtype=float)
    sources, targets = edges.T
    distances = state.edge_distances if state.opinions.dtype == x.dtype else np.abs(x[sources] - x[targets])
    concordant = distances < epsilon
    degree = state.out_degree
    close = np.bincount(sources, weights=concordant, minlength=len(degree))
    valid = degree > 0
    return float(np.mean(close[valid] / degree[valid]))
