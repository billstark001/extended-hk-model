"""Baseline cache provenance and descriptor/interpolant exports."""

from __future__ import annotations

import csv
import json

import numpy as np

from common.numerics.interpolation import pchip
from experiments.paper_theory.energy_scan.baseline import config


def save_descriptor_csv(rows: list[dict[str, object]]) -> None:
    """Write baseline energy descriptor rows to the configured generated directory."""
    with (config.HERE / "curve_parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_interpolants(rows: list[dict[str, object]], quantiles: np.ndarray) -> None:
    """Persist interpolation-ready descriptor/quantile data on the declared confidence/rate grid."""
    scalar_names = (
        "plateau",
        "drop_amplitude",
        "first_step_fraction",
        "initial_speed_tau",
        "tau_10",
        "tau_25",
        "tau_50",
        "tau_75",
        "tau_90",
        "tau_99",
        "duration_10_90",
        "shape_25",
        "shape_50",
        "shape_75",
        "tail_slowdown",
    )
    archive: dict[str, np.ndarray] = {
        "epsilon": config.EPSILONS,
        "q_over_alpha": config.RATIOS,
        "positive_log1p_q_over_alpha": np.log1p(config.RATIOS[1:]),
        "progress_levels": config.PROGRESS_LEVELS,
        "tau_quantiles": quantiles,
    }
    for epsilon_index, epsilon in enumerate(config.EPSILONS):
        for energy_index, energy in enumerate(config.ENERGY_NAMES):
            selected = [row for row in rows if row["epsilon"] == float(epsilon) and row["energy"] == energy]
            prefix = f"e{epsilon:g}_{energy}"
            x = np.log1p(config.RATIOS[1:])
            for name in scalar_names:
                values = np.asarray([float(row[name]) for row in selected])[1:]
                finite = np.isfinite(values)
                if np.count_nonzero(finite) < 2:
                    continue
                interpolator = pchip(x[finite], values[finite], extrapolate=False)
                archive[f"{prefix}_{name}_knots"] = interpolator.x
                archive[f"{prefix}_{name}_coefficients"] = interpolator.c
            for level_index, level in enumerate(config.PROGRESS_LEVELS):
                values = quantiles[epsilon_index, 1:, energy_index, level_index]
                finite = np.isfinite(values)
                if np.count_nonzero(finite) < 2:
                    continue
                interpolator = pchip(x[finite], values[finite], extrapolate=False)
                label = str(round(100 * level)).zfill(2)
                archive[f"{prefix}_tau_{label}_knots"] = interpolator.x
                archive[f"{prefix}_tau_{label}_coefficients"] = interpolator.c
    np.savez_compressed(config.HERE / "descriptor_interpolants.npz", **archive)


def save_data(data: dict[str, np.ndarray], audit: dict[str, np.ndarray]) -> None:
    """Persist baseline trajectory arrays with their generation protocol."""
    from experiments.paper_theory.energy_scan.baseline.design import protocol

    np.savez_compressed(
        config.HERE / "energy_curve_parameterization.npz",
        protocol=np.asarray(json.dumps(protocol(), sort_keys=True)),
        epsilon=config.EPSILONS,
        q_over_alpha=config.RATIOS,
        q=config.ALPHA * config.RATIOS,
        alpha=np.asarray(config.ALPHA),
        progress_levels=config.PROGRESS_LEVELS,
        **data,
        **audit,
    )
