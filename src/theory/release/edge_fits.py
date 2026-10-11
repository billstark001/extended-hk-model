"""Profiled multistart fast-edge/node-survival mixture regression."""

import math

import numpy as np
from scipy.optimize import least_squares


def fit_edge_curve(tau, node_survival, edge_survival, ratio, beta_bounds=(0.2, 5.0)):
    log_kappa_bounds = (math.log(0.0001), math.log(1000.0))
    log_beta_bounds = tuple((math.log(value) for value in beta_bounds))
    log_kappa_grid = np.linspace(*log_kappa_bounds, 161)
    beta_starts = np.asarray((0.3, 0.5, 0.8, 1.0, 1.5, 2.5, 4.5))
    if ratio == 0.0:
        weight = 0.0
        kappa = float("nan")
        beta = float("nan")
        prediction = node_survival
        optimizer_success = True
        jacobian_condition = float("nan")
    else:
        sample_count = min(1501, tau.size)
        sample_index = np.unique(np.linspace(0, tau.size - 1, sample_count).astype(int))
        fit_tau = tau[sample_index]
        fit_node = node_survival[sample_index]
        fit_edge = edge_survival[sample_index]

        def profiled_fit(log_kappa: float, log_beta: float) -> tuple[float, float]:
            scaled_time = np.exp(log_kappa) * ratio * fit_tau
            exponential = np.exp(-np.power(scaled_time, np.exp(log_beta)))
            direction = exponential - fit_node
            denominator = float(direction @ direction)
            if denominator <= 1e-20:
                trial_weight = 0.0
            else:
                trial_weight = float(np.clip((fit_edge - fit_node) @ direction / denominator, 0.0, 1.0))
            residual = fit_node + trial_weight * direction - fit_edge
            return (float(np.mean(residual**2)), trial_weight)

        starts: list[tuple[float, float, float]] = []
        for beta_start in beta_starts:
            log_beta_start = math.log(float(beta_start))
            costs = np.asarray([profiled_fit(float(log_kappa), log_beta_start)[0] for log_kappa in log_kappa_grid])
            best_index = int(np.argmin(costs))
            log_kappa_start = float(log_kappa_grid[best_index])
            _, weight_start = profiled_fit(log_kappa_start, log_beta_start)
            starts.append((weight_start, log_kappa_start, log_beta_start))

        def residual(parameters: np.ndarray) -> np.ndarray:
            trial_weight = float(parameters[0])
            scaled_time = np.exp(parameters[1]) * ratio * fit_tau
            exponential = np.exp(-np.power(scaled_time, np.exp(parameters[2])))
            return fit_node + trial_weight * (exponential - fit_node) - fit_edge

        fitted = [
            least_squares(
                residual,
                np.asarray(start),
                bounds=(
                    np.asarray((0.0, log_kappa_bounds[0], log_beta_bounds[0])),
                    np.asarray((1.0, log_kappa_bounds[1], log_beta_bounds[1])),
                ),
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=1200,
            )
            for start in starts
        ]
        best_fit = min(fitted, key=lambda fit: float(fit.cost))
        weight = float(best_fit.x[0])
        kappa = float(np.exp(best_fit.x[1]))
        beta = float(np.exp(best_fit.x[2]))
        optimizer_success = bool(best_fit.success)
        singular_values = np.linalg.svd(best_fit.jac, compute_uv=False)
        jacobian_condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size == 3 and singular_values[-1] > 1e-12
            else float("inf")
        )
        prediction = weight * np.exp(-np.power(kappa * ratio * tau, beta)) + (1.0 - weight) * node_survival
    error = prediction - edge_survival
    rms_error = float(np.sqrt(np.mean(error**2)))
    return dict(
        weight=weight,
        kappa=kappa,
        beta=beta,
        optimizer_success=optimizer_success,
        jacobian_condition=jacobian_condition,
        rms_error=rms_error,
        error=error,
    )
