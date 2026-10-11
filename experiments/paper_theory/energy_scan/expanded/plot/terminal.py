"""Terminal energy, potential-well and derivative figures from saved fields."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def plot_terminal_energies_and_wells(rows: list[dict[str, Any]]) -> None:
    """Render terminal energies and potential-well descriptors from archived final states."""
    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure
    from experiments.paper_theory.energy_scan.expanded.plot.design import selected_parameter_rows

    figure, axes = plt.subplots(4, 3, figsize=(11.0, 11.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, config.ALPHAS.size))
    for dynamics_index, dynamics in enumerate(config.DYNAMICS):
        energy_row = 2 * dynamics_index
        well_row = energy_row + 1
        for epsilon_index, epsilon in enumerate(config.EPSILONS):
            energy_axis = axes[energy_row, epsilon_index]
            well_axis = axes[well_row, epsilon_index]
            for alpha, color in zip(config.ALPHAS, colors, strict=True):
                subset = selected_parameter_rows(rows, dynamics, float(epsilon), float(alpha))
                ratio = np.asarray([row["q_over_alpha"] for row in subset])
                node_energy = np.asarray([row["node_final_energy"] for row in subset])
                edge_energy = np.asarray([row["edge_final_energy"] for row in subset])
                well_count = np.asarray([row["node_potential_well_count"] for row in subset])
                converged = np.asarray([row["trajectory_converged"] for row in subset])
                energy_axis.plot(ratio, node_energy, color=color, marker="o", ms=2, lw=0.9)
                energy_axis.plot(ratio, edge_energy, color=color, marker="o", ms=2, lw=0.9, ls="--")
                well_axis.plot(ratio, well_count, color=color, marker="o", ms=2, lw=0.9)
                if np.any(~converged):
                    for axis, values in (
                        (energy_axis, node_energy),
                        (energy_axis, edge_energy),
                        (well_axis, well_count),
                    ):
                        axis.scatter(
                            ratio[~converged],
                            values[~converged],
                            s=18,
                            facecolor="white",
                            edgecolor=color,
                            linewidth=0.9,
                            zorder=3,
                        )
            energy_axis.set_xscale("log")
            energy_axis.set_yscale("log")
            energy_axis.grid(alpha=0.2, which="both")
            well_axis.set_xscale("log")
            well_axis.set_yticks(np.arange(1, 6))
            well_axis.set_ylim(0.7, 5.3)
            well_axis.grid(alpha=0.2, which="both")
            if energy_row == 0:
                energy_axis.set_title(f"$\\epsilon={epsilon:g}$")
            if epsilon_index == 0:
                energy_axis.set_ylabel(f"{dynamics.upper()} final energy")
                well_axis.set_ylabel(f"{dynamics.upper()} potential wells")
            well_axis.set_xlabel("$q/\\alpha$")
    alpha_handles = [
        plt.Line2D((), (), color=color, label=f"$\\alpha={alpha:.4g}$")
        for alpha, color in zip(config.ALPHAS, colors, strict=True)
    ]
    quantity_handles = [
        plt.Line2D((), (), color="black", label="node $c_\\rho$"),
        plt.Line2D((), (), color="black", ls="--", label="edge $c_E$"),
        plt.Line2D(
            (),
            (),
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="none",
            label="right-censored at $\\tau=120$",
        ),
    ]
    axes[0, -1].legend(handles=alpha_handles + quantity_handles, fontsize=7, frameon=False)
    figure.suptitle("Terminal interaction energies and node-potential well counts", y=1.005)
    save_figure(figure, "f_terminal_energies_and_wells")
