"""Balanced initial-state/process variance components in scalar or weighted field norms."""

import numpy as np


def components(y, prediction, weight=1.0):
    """Return W, corrected I/B² and the two finite-sample algebraic identities.

    y has shape (initials, processes, features); prediction has shape
    (initials, features). A scalar can be represented by one feature. The norm is
    weight * sum(features**2); corrected estimates deliberately retain negatives.
    """
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 3 or prediction.shape != (y.shape[0], y.shape[2]) or min(y.shape[:2]) < 2:
        raise ValueError("balanced data need >=2 independent initials and >=2 processes per initial")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)):
        raise ValueError("nonfinite population observations")
    count, repeats = y.shape[:2]
    means = y.mean(1)
    residual = means - prediction

    def norm(value):
        return np.sum(value * value, axis=-1) * weight

    def variance(value):
        return norm(value - value.mean(0)).sum() / (count - 1)

    within = norm(y - means[:, None]).sum() / (count * (repeats - 1))
    mean_error = norm(residual).mean()
    covariance = weight * np.sum((prediction - prediction.mean(0)) * (residual - residual.mean(0))) / (count - 1)
    result = {
        "process_variance": float(within),
        "initial_variance_corrected": float(variance(means) - within / repeats),
        "conditional_bias_squared_corrected": float(mean_error - within / repeats),
        "conditional_mean_error_squared": float(mean_error),
        "individual_mse": float(norm(y - prediction[:, None]).mean()),
        "prediction_initial_variance": float(variance(prediction)),
        "residual_initial_variance_corrected": float(variance(residual) - within / repeats),
        "prediction_residual_covariance": float(covariance),
    }
    if y.shape[-1] == 1:
        result["signed_bias"] = float(residual.mean())
    return result


def with_loio(y, prediction, weight=1.0):
    """Attach leave-one-initial-out extrema, which are sensitivity ranges, not CIs."""
    result = components(y, prediction, weight)
    alternatives = (
        [components(np.delete(y, g, axis=0), np.delete(prediction, g, axis=0), weight) for g in range(len(y))]
        if len(y) >= 3
        else []
    )
    for key in ("process_variance", "initial_variance_corrected", "conditional_bias_squared_corrected"):
        result[key + "_loio_min"] = min(r[key] for r in alternatives) if alternatives else np.nan
        result[key + "_loio_max"] = max(r[key] for r in alternatives) if alternatives else np.nan
    return result
