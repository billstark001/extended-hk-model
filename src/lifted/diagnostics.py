"""Joint increment factors avoid a quadratic covariance matrix in the state dimension."""

import numpy as np


def increment_factors(increments, *, bins, degree):
    """Return mean and (samples, B+B²) factor F with covariance F.T@F.

    Node/edge entries are cell masses per individual. Every sample must preserve
    mass and edge row sums jointly. No dense covariance is materialized.
    """
    x = np.asarray(increments, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != bins + bins * bins or not np.isfinite(x).all():
        raise ValueError("at least two finite joint increments with B+B² columns required")
    rho, edge = x[:, :bins], x[:, bins:].reshape(-1, bins, bins)
    if np.max(abs(rho.sum(axis=1))) > 1e-9 or np.max(abs(edge.sum(axis=2) - degree * rho)) > 1e-9:
        raise ValueError("joint mass/fixed-degree increment constraint violated")
    mean = x.mean(axis=0)
    factor = (x - mean) / np.sqrt(len(x) - 1)
    return mean, factor


def factor_summary(mean, factor, bins):
    """Trace and cross-block Frobenius norm computed through the sample Gram matrices."""
    f = np.asarray(factor)
    node, edge = f[:, :bins], f[:, bins:]
    cross_squared = float(np.sum((node @ node.T) * (edge @ edge.T)))
    return dict(
        node_variance_trace=float(np.sum(node * node)),
        edge_variance_trace=float(np.sum(edge * edge)),
        cross_covariance_norm=float(np.sqrt(max(0, cross_squared))),
        drift_norm=float(np.linalg.norm(mean)),
    )
