"""Low-parameter fits to release quantiles; fitting time coordinates are scaled tau."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def quantile_times(row: dict[str, Any]) -> np.ndarray:
    """Read configured tau quantiles from one descriptor row in progress-level order."""
    return np.asarray([float(row[f"tau_{round(100 * level):02d}"]) for level in config.PROGRESS_LEVELS])


from theory.release.quantile_fits import fit_exponentiated_weibull, fit_gompertz, fit_rational_quadratic


def low_parameter_fits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit all declared low-parameter models to valid descriptor rows, skipping nonfinite/nonpositive quantile times."""
    models = {
        "gompertz_hazard_2": (fit_gompertz, ("initial_hazard", "acceleration")),
        "exponentiated_weibull_3": (fit_exponentiated_weibull, ("scale", "beta", "gamma")),
        "rational_quadratic_bezier_3": (fit_rational_quadratic, ("scale", "control", "weight")),
    }
    output = []
    for row in rows:
        times = quantile_times(row)
        if np.any(~np.isfinite(times)) or np.any(times <= 0):
            continue
        for model, (fit_model, names) in models.items():
            prediction, parameters = fit_model(times, config.PROGRESS_LEVELS)
            log_error = np.log(prediction / times)
            record = {key: row[key] for key in ("dynamics", "epsilon", "alpha", "q", "q_over_alpha", "energy")}
            record.update(
                {
                    "model": model,
                    "rms_log_time_error": float(np.sqrt(np.mean(log_error**2))),
                    "median_absolute_relative_error": float(np.median(np.abs(prediction / times - 1))),
                    "maximum_absolute_relative_error": float(np.max(np.abs(prediction / times - 1))),
                    **{name: float(value) for name, value in zip(names, parameters, strict=True)},
                }
            )
            output.append(record)
    return output
