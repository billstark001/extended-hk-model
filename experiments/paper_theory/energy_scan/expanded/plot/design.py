"""Feasible-rate design, relaxation-time panels and ordered plot-row selection."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def plot_design() -> None:
    """Save the positive-rate triangular grid with q=alpha*ratio <= 1 on logarithmic axes."""
    from experiments.paper_theory.energy_scan.expanded.design import feasible_ratios
    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    for alpha in config.ALPHAS:
        ratios = feasible_ratios(float(alpha))[1:]
        axis.scatter(ratios, np.full_like(ratios, alpha), s=18, color="#31688e")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("positive relative rate $q/\\alpha$")
    axis.set_ylabel("influence rate $\\alpha$")
    axis.set_title("Log-uniform triangular design; each right edge has $q=1$")
    axis.grid(alpha=0.2, which="both")
    save_figure(figure, "f_design_grid")


def plot_relaxation_times(rows: list[dict[str, Any]]) -> None:
    """Save scaled T50/T90 versus rate-ratio panels for the configured dynamics, confidence and alpha grid."""
    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    figure, axes = plt.subplots(4, 3, figsize=(11.0, 11.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, config.ALPHAS.size))
    for dynamics_index, dynamics in enumerate(config.DYNAMICS):
        for energy_index, energy in enumerate(config.ENERGY_NAMES):
            row_index = 2 * dynamics_index + energy_index
            for epsilon_index, epsilon in enumerate(config.EPSILONS):
                axis = axes[row_index, epsilon_index]
                for alpha, color in zip(config.ALPHAS, colors, strict=True):
                    subset = selected_rows(rows, dynamics, float(epsilon), float(alpha), energy)
                    ratio = np.asarray([row["q_over_alpha"] for row in subset])
                    axis.plot(ratio, [row["tau_50"] for row in subset], color=color, marker="o", ms=2, lw=0.9)
                    axis.plot(ratio, [row["tau_90"] for row in subset], color=color, ls="--", lw=0.9)
                axis.set_xscale("log")
                axis.grid(alpha=0.2, which="both")
                if row_index == 0:
                    axis.set_title(f"$\\epsilon={epsilon:g}$")
                if epsilon_index == 0:
                    label = "$\\mathcal{U}_\\rho$" if energy_index == 0 else "$\\mathcal{U}_E$"
                    axis.set_ylabel(f"{dynamics.upper()} {label}\nscaled time")
                if row_index == 3:
                    axis.set_xlabel("$q/\\alpha$")
    alpha_handles = [
        plt.Line2D((), (), color=color, label=f"$\\alpha={alpha:.4g}$")
        for alpha, color in zip(config.ALPHAS, colors, strict=True)
    ]
    time_handles = [
        plt.Line2D((), (), color="black", label="$T_{50}$"),
        plt.Line2D((), (), color="black", ls="--", label="$T_{90}$"),
    ]
    axes[0, -1].legend(handles=alpha_handles + time_handles, fontsize=7, frameon=False)
    figure.suptitle("Relaxation times over the extended log-uniform scan", y=1.005)
    save_figure(figure, "f_relaxation_times")


def selected_parameter_rows(
    rows: list[dict[str, Any]], dynamics: str, epsilon: float, alpha: float
) -> list[dict[str, Any]]:
    """Return positive-rate rows matching dynamics/confidence/alpha, sorted by rate ratio."""
    return sorted(
        (
            row
            for row in rows
            if row["dynamics"] == dynamics
            and row["epsilon"] == epsilon
            and np.isclose(row["alpha"], alpha)
            and (row["q_over_alpha"] > 0)
        ),
        key=lambda row: row["q_over_alpha"],
    )


def selected_rows(
    rows: list[dict[str, Any]], dynamics: str, epsilon: float, alpha: float, energy: str
) -> list[dict[str, Any]]:
    """Return positive-rate rows matching dynamics/confidence/alpha/energy, sorted by rate ratio."""
    return sorted(
        (
            row
            for row in rows
            if row["dynamics"] == dynamics
            and row["epsilon"] == epsilon
            and np.isclose(row["alpha"], alpha)
            and (row["energy"] == energy)
            and (row["q_over_alpha"] > 0)
        ),
        key=lambda row: row["q_over_alpha"],
    )
