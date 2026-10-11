"""Baseline release descriptors, converting physical time to scaled tau explicitly."""

from __future__ import annotations

import numpy as np

from experiments.paper_theory.energy_scan.baseline import config
from theory.release.descriptors import curve_parameters


def parameterize(data: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], np.ndarray]:
    """Return descriptor rows and an (epsilon, ratio, energy, progress) quantile tensor from padded baseline arrays."""
    rows: list[dict[str, object]] = []
    quantiles = np.full(
        (config.EPSILONS.size, config.RATIOS.size, len(config.ENERGY_NAMES), config.PROGRESS_LEVELS.size), np.nan
    )
    for epsilon_index, epsilon in enumerate(config.EPSILONS):
        for ratio_index, ratio in enumerate(config.RATIOS):
            length = int(data["lengths"][epsilon_index, ratio_index])
            time = data["time"][epsilon_index, ratio_index, :length]
            for energy_index, name in enumerate(config.ENERGY_NAMES):
                values = data[name][epsilon_index, ratio_index, :length]
                result = curve_parameters(time, values, config.ALPHA, config.PROGRESS_LEVELS)
                quantiles[epsilon_index, ratio_index, energy_index] = result.pop("tau_quantiles")
                result.pop("progress")
                rows.append(
                    {
                        "epsilon": float(epsilon),
                        "alpha": config.ALPHA,
                        "q": float(config.ALPHA * ratio),
                        "q_over_alpha": float(ratio),
                        "energy": name,
                        "executed_steps": int(data["executed_steps"][epsilon_index, ratio_index]),
                        "converged": bool(data["converged"][epsilon_index, ratio_index]),
                        **result,
                    }
                )
    return (rows, quantiles)
