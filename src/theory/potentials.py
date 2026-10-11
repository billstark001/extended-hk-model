"""Diagnostics of model interaction potential wells."""

import numpy as np
from scipy.signal import find_peaks


def count_potential_wells(potential: np.ndarray, prominence_fraction: float = 0.02) -> int:
    """Count robust interior minima after five-cell binomial smoothing."""
    values = np.asarray(potential, dtype=float)
    if np.any(~np.isfinite(values)):
        return 0
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
    smooth = np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")
    value_range = float(np.ptp(smooth))
    if value_range <= 1e-14:
        return 0
    minima, _ = find_peaks(-smooth, prominence=prominence_fraction * value_range, distance=3)
    return int(minima.size)
