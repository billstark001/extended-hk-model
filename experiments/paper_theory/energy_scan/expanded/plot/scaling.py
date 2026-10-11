"""Influence-rate collapse and release-quantile compression error figures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def plot_alpha_collapse(shards: dict[tuple[str, float], dict[str, np.ndarray]]) -> None:
    """Render scaled-time energy collapse across influence values at matching physical ratios."""
    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    target_ratio = 100.0
    figure, axes = plt.subplots(4, 3, figsize=(11.0, 11.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, config.ALPHAS.size))
    for dynamics_index, dynamics in enumerate(config.DYNAMICS):
        for energy_index, energy in enumerate(config.ENERGY_NAMES):
            row_index = 2 * dynamics_index + energy_index
            for epsilon_index, epsilon in enumerate(config.EPSILONS):
                axis = axes[row_index, epsilon_index]
                for alpha, color in zip(config.ALPHAS, colors, strict=True):
                    shard = shards[dynamics, float(alpha)]
                    ratio_index = int(np.flatnonzero(np.isclose(shard["q_over_alpha"], target_ratio))[0])
                    length = int(shard["lengths"][epsilon_index, ratio_index])
                    tau = shard["tau"][epsilon_index, ratio_index, :length]
                    values = shard[energy][epsilon_index, ratio_index, :length]
                    tail = min(501, max(51, length // 10))
                    plateau = np.median(values[-tail:])
                    survival = np.clip((values - plateau) / (values[0] - plateau), 0.0, 1.0)
                    axis.plot(tau, survival, color=color, lw=1)
                axis.set_xlim(0, 45)
                axis.grid(alpha=0.2)
                if row_index == 0:
                    axis.set_title(f"$\\epsilon={epsilon:g}$")
                if epsilon_index == 0:
                    label = "$S_\\rho$" if energy_index == 0 else "$S_E$"
                    axis.set_ylabel(f"{dynamics.upper()} {label}")
                if row_index == 3:
                    axis.set_xlabel("scaled time $\\tau=\\alpha t$")
    axes[0, -1].legend(
        [
            plt.Line2D((), (), color=color, label=f"$\\alpha={alpha:.4g}$")
            for alpha, color in zip(config.ALPHAS, colors, strict=True)
        ],
        [f"$\\alpha={alpha:.4g}$" for alpha in config.ALPHAS],
        fontsize=7,
        frameon=False,
    )
    figure.suptitle("Finite-step collapse at fixed $q/\\alpha=100$", y=1.005)
    save_figure(figure, "f_alpha_collapse_r100")


def plot_model_errors(fits: list[dict[str, Any]]) -> None:
    """Save model quantile-compression error summaries across dynamics and energy channels."""
    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    models = ("gompertz_hazard_2", "exponentiated_weibull_3", "rational_quadratic_bezier_3")
    labels = ("Gompertz (2)", "Exp. Weibull (3)", "Rational Bézier (3)")
    figure, axes = plt.subplots(2, 2, figsize=(9.3, 6.8), sharey=True)
    for dynamics_index, dynamics in enumerate(config.DYNAMICS):
        for energy_index, energy in enumerate(config.ENERGY_NAMES):
            axis = axes[dynamics_index, energy_index]
            medians = []
            upper = []
            for model in models:
                values = np.asarray(
                    [
                        row["maximum_absolute_relative_error"]
                        for row in fits
                        if row["dynamics"] == dynamics and row["energy"] == energy and (row["model"] == model)
                    ]
                )
                medians.append(np.median(values))
                upper.append(np.quantile(values, 0.9))
            x = np.arange(len(models))
            axis.bar(x, medians, color=("#4477aa", "#66ccee", "#228833"))
            axis.scatter(x, upper, marker="_", s=160, color="black", label="90th percentile")
            axis.set_xticks(x, labels, rotation=18, ha="right")
            axis.set_yscale("log")
            axis.grid(alpha=0.2, axis="y", which="both")
            axis.set_title(
                f"{dynamics.upper()} — " + ("$\\mathcal{U}_\\rho$" if energy_index == 0 else "$\\mathcal{U}_E$")
            )
            if energy_index == 0:
                axis.set_ylabel("maximum quantile-time relative error")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle("Low-parameter compression audit", y=1.01)
    save_figure(figure, "f_low_parameter_model_audit")
