"""Small-update linearity and fixed-ratio curve-collapse diagnostics."""

from __future__ import annotations

import json

import numpy as np

from experiments.paper_theory.energy_scan.scaling import config


def through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit y=slope*x through the origin for channel probe data."""
    return float(np.dot(x, y) / np.dot(x, x))


def r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Return variance-relative prediction quality for observed versus fitted values."""
    residual = np.sum((observed - fitted) ** 2)
    total = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - residual / total)


def collapse_error(values: np.ndarray) -> dict[str, float | None]:
    """Return the largest finite gap against values[0] and its normalization by the reference maximum."""
    reference = values[0]
    reference_all = np.broadcast_to(reference, values.shape)
    valid = np.isfinite(values) & np.isfinite(reference_all)
    if not np.any(valid):
        return {"max_absolute": None, "relative_to_reference_max": None}
    difference = float(np.max(np.abs(values[valid] - reference_all[valid])))
    scale = float(np.max(np.abs(reference_all[valid])))
    return {
        "max_absolute": difference,
        "relative_to_reference_max": float(difference / scale) if scale > 1e-12 else None,
    }


def build_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Write channel coefficients, first-step scaling and dynamic/field collapse diagnostics for the cached dataset."""
    alpha_dissipation = -data["alpha_delta_u"]
    q_dissipation = -data["q_delta_ue"]
    small = data["alpha_probe"] <= 0.0316227767
    log_slope = np.polyfit(np.log(data["alpha_probe"][small]), np.log(alpha_dissipation[small]), 1)[0]
    q_slope = through_origin_slope(data["q_probe"], q_dissipation)
    alpha_prediction = float(data["alpha_coefficient_continuum"])
    q_prediction = float(data["q_coefficient_continuum"])
    dynamic: dict[str, object] = {}
    for gamma_index, gamma in enumerate(data["gamma_collapse"]):
        key = f"gamma_{gamma:g}"
        dynamic[key] = {
            "U": collapse_error(data["energy_rho"][gamma_index]),
            "U_E": collapse_error(data["energy_edge"][gamma_index]),
            "Delta_U_top": collapse_error(data["topology_excess"][gamma_index]),
            "Phi_rho_final": collapse_error(data["phi_rho"][gamma_index, :, -1]),
            "Phi_E_final": collapse_error(data["phi_edge"][gamma_index, :, -1]),
            "Delta_U_top_final_by_alpha": data["topology_excess"][gamma_index, :, -1].tolist(),
            "maximum_positive_U_increment": float(
                np.max(np.maximum(np.diff(data["energy_rho"][gamma_index], axis=1), 0.0))
            ),
            "maximum_positive_U_E_increment": float(
                np.max(np.maximum(np.diff(data["energy_edge"][gamma_index], axis=1), 0.0))
            ),
        }
    summary: dict[str, object] = {
        "protocol": json.loads(str(data["protocol"].item())),
        "frozen_channel_tests": {
            "small_alpha_loglog_slope": float(log_slope),
            "continuum_minus_dU_over_alpha": alpha_prediction,
            "smallest_alpha_minus_dU_over_alpha": float(alpha_dissipation[0] / data["alpha_probe"][0]),
            "smallest_alpha_relative_coefficient_error": float(
                abs(alpha_dissipation[0] / data["alpha_probe"][0] - alpha_prediction) / alpha_prediction
            ),
            "q_fit_through_origin_slope": q_slope,
            "continuum_minus_dUE_over_q": q_prediction,
            "q_slope_relative_error": float(abs(q_slope - q_prediction) / q_prediction),
            "q_linearity_R2": r_squared(q_dissipation, q_slope * data["q_probe"]),
            "max_abs_q_topology_identity_error": float(np.max(np.abs(data["q_delta_topology"] - data["q_delta_ue"]))),
        },
        "dynamic_scaled_time_collapse": dynamic,
        "q_zero_product_identity_max_abs": float(np.max(np.abs(data["topology_excess"][0]))),
        "maximum_solver_node_mass_residual": float(np.nanmax(data["max_node_residual"])),
        "maximum_solver_degree_residual": float(np.nanmax(data["max_degree_residual"])),
        "uniform_epsilon_test": {
            "max_abs_U_error": float(np.max(np.abs(data["epsilon_numerical_u"] - data["epsilon_analytic_u"]))),
            "max_abs_U_E_error": float(np.max(np.abs(data["epsilon_numerical_ue"] - data["epsilon_analytic_ue"]))),
            "max_abs_dU_depsilon_error": float(
                np.max(np.abs(data["epsilon_numerical_du"] - data["epsilon_analytic_du"]))
            ),
            "max_abs_dU_E_depsilon_error": float(
                np.max(np.abs(data["epsilon_numerical_due"] - data["epsilon_analytic_due"]))
            ),
            "relative_U_error_at_epsilon_0p45": float(
                abs(
                    np.interp(config.EPSILON, data["epsilon_probe"], data["epsilon_numerical_u"])
                    - np.interp(config.EPSILON, data["epsilon_probe"], data["epsilon_analytic_u"])
                )
                / np.interp(config.EPSILON, data["epsilon_probe"], data["epsilon_analytic_u"])
            ),
        },
    }
    (config.HERE / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
