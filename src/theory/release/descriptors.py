"""Historical scan release descriptors; caller supplies its physical scales/levels."""

import numpy as np


def crossing_time(time: np.ndarray, progress: np.ndarray, level: float) -> float:
    """Return an interpolated first progress crossing or NaN when it lies beyond the observed window."""
    reached = np.flatnonzero(progress >= level)
    if reached.size == 0:
        return float("nan")
    right = int(reached[0])
    if right == 0:
        return float(time[0])
    left = right - 1
    change = progress[right] - progress[left]
    if change <= 1e-15:
        return float(time[right])
    fraction = (level - progress[left]) / change
    return float(time[left] + fraction * (time[right] - time[left]))


def curve_parameters(
    time: np.ndarray, values: np.ndarray, alpha: float, levels: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Return tail-normalized descriptors and quantiles; input time is physical and returned quantile times use alpha*time."""
    tail_count = min(501, max(51, values.size // 10))
    plateau = float(np.median(values[-tail_count:]))
    monotone = np.minimum.accumulate(values)
    amplitude = float(values[0] - plateau)
    if amplitude <= 1e-14:
        progress = np.zeros_like(values)
        quantiles = np.full(levels.size, np.nan)
    else:
        progress = np.maximum.accumulate(np.clip((values[0] - monotone) / amplitude, 0.0, 1.0))
        quantiles = np.asarray([crossing_time(time, progress, level) for level in levels])
    tau_quantiles = alpha * quantiles
    lookup = {float(level): tau_quantiles[index] for index, level in enumerate(levels)}
    duration = lookup[0.9] - lookup[0.1]
    first_drop = float(values[0] - values[1]) if values.size > 1 else float("nan")
    rates = -np.diff(values) / (alpha * np.diff(time)) if values.size > 1 else np.asarray((np.nan,))
    return {
        "initial_energy": float(values[0]),
        "plateau": plateau,
        "drop_amplitude": amplitude,
        "first_step_drop": first_drop,
        "first_step_fraction": first_drop / amplitude if amplitude > 1e-14 else float("nan"),
        "initial_speed_tau": first_drop / alpha,
        "maximum_speed_tau": float(np.nanmax(rates)),
        "nonincreasing_fraction": float(np.mean(np.diff(values) <= 1e-12)),
        "tau_05": lookup[0.05],
        "tau_10": lookup[0.1],
        "tau_25": lookup[0.25],
        "tau_50": lookup[0.5],
        "tau_75": lookup[0.75],
        "tau_90": lookup[0.9],
        "tau_95": lookup[0.95],
        "tau_99": lookup[0.99],
        "duration_10_90": duration,
        "shape_25": (lookup[0.25] - lookup[0.1]) / duration if duration > 0 else float("nan"),
        "shape_50": (lookup[0.5] - lookup[0.1]) / duration if duration > 0 else float("nan"),
        "shape_75": (lookup[0.75] - lookup[0.1]) / duration if duration > 0 else float("nan"),
        "tail_slowdown": (lookup[0.99] - lookup[0.9]) / duration if duration > 0 else float("nan"),
        "progress": progress,
        "tau_quantiles": tau_quantiles,
    }


def curve_descriptors(
    tau: np.ndarray, values: np.ndarray, first_step_delta: float, alpha: float, levels: np.ndarray
) -> dict[str, float]:
    """Return plateau, amplitude, release quantiles and shape/speed diagnostics from one tau/energy curve."""
    tail_count = min(501, max(51, values.size // 10))
    plateau = float(np.median(values[-tail_count:]))
    monotone = np.minimum.accumulate(values)
    amplitude = float(values[0] - plateau)
    if amplitude <= 1e-14:
        quantiles = np.full(levels.size, np.nan)
    else:
        progress = np.maximum.accumulate(np.clip((values[0] - monotone) / amplitude, 0.0, 1.0))
        quantiles = np.asarray([crossing_time(tau, progress, level) for level in levels])
    lookup = dict(zip(levels, quantiles, strict=True))
    duration = lookup[0.9] - lookup[0.1]
    rates = -np.diff(values) / np.diff(tau)
    first_drop = -float(first_step_delta)
    return {
        "initial_energy": float(values[0]),
        "plateau": plateau,
        "drop_amplitude": amplitude,
        "first_step_drop": first_drop,
        "first_step_fraction": first_drop / amplitude if amplitude > 1e-14 else float("nan"),
        "initial_speed_tau": first_drop / alpha,
        "maximum_speed_tau": float(np.nanmax(rates)),
        "nonincreasing_fraction": float(np.mean(np.diff(values) <= 1e-12)),
        **{f"tau_{round(100 * level):02d}": float(value) for level, value in zip(levels, quantiles, strict=True)},
        "duration_10_90": duration,
        "shape_25": (lookup[0.25] - lookup[0.1]) / duration,
        "shape_50": (lookup[0.5] - lookup[0.1]) / duration,
        "shape_75": (lookup[0.75] - lookup[0.1]) / duration,
        "tail_slowdown": (lookup[0.99] - lookup[0.9]) / duration,
    }
