"""Render baseline trajectories and descriptors, using tau=alpha*t for time axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.energy_scan.baseline import config


def nearest_ratio(value: float) -> int:
    """Return the configured rate-ratio index nearest to value."""
    return int(np.argmin(np.abs(config.RATIOS - value)))


def plot_energy_curves(data: dict[str, np.ndarray]) -> None:
    """Save node/edge trajectories at selected rate ratios, with energy divided by epsilon squared."""
    selected = [nearest_ratio(value) for value in (0, 1, 10, 100)]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected)))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), sharex=True)
    for column, epsilon in enumerate(config.EPSILONS):
        for row, name in enumerate(config.ENERGY_NAMES):
            axis = axes[row, column]
            for index, color in zip(selected, colors, strict=True):
                length = int(data["lengths"][column, index])
                tau = config.ALPHA * data["time"][column, index, :length]
                values = data[name][column, index, :length] / epsilon**2
                axis.plot(tau, values, color=color, lw=1.2, label=f"$q/\\alpha={config.RATIOS[index]:g}$")
            axis.grid(alpha=0.2)
            axis.set_xlim(0, 45)
            if row == 0:
                axis.set_title(f"$\\epsilon={epsilon:g}$")
            if column == 0:
                axis.set_ylabel("$\\mathcal{U}_\\rho/\\epsilon^2$" if row == 0 else "$\\mathcal{U}_E/\\epsilon^2$")
            if row == 1:
                axis.set_xlabel("scaled time $\\tau=\\alpha t$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Energy relaxation at fixed influence and sliding rewiring rate", y=1.01)
    save_figure(fig, "f_energy_curve_scan")


def row_matrix(rows: list[dict[str, object]], energy: str, name: str) -> np.ndarray:
    """Arrange one descriptor as an (epsilon, ratio) array on the configured complete baseline grid."""
    return np.asarray(
        [
            [
                float(
                    next(
                        (
                            row[name]
                            for row in rows
                            if row["epsilon"] == float(epsilon)
                            and row["energy"] == energy
                            and (row["q_over_alpha"] == float(ratio))
                        )
                    )
                )
                for ratio in config.RATIOS
            ]
            for epsilon in config.EPSILONS
        ]
    )


def plot_descriptors(rows: list[dict[str, object]]) -> None:
    """Save release-time, shape and amplitude panels from complete baseline descriptor rows."""
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 10.4), sharex=True)
    colors = {"tau_50": "#1f77b4", "tau_90": "#ff7f0e", "tau_99": "#2ca02c"}
    for column, epsilon in enumerate(config.EPSILONS):
        for row_index, energy in enumerate(config.ENERGY_NAMES):
            axis = axes[row_index, column]
            for name, color in colors.items():
                values = row_matrix(rows, energy, name)[column]
                axis.plot(
                    config.RATIOS,
                    values,
                    "o-",
                    ms=2.5,
                    lw=1,
                    color=color,
                    label=name.replace("tau_", "$\\tau_{") + "}$",
                )
            axis.set_xscale("symlog", linthresh=0.25)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(f"$\\epsilon={epsilon:g}$")
            if column == 0:
                axis.set_ylabel("node times $\\tau_p$" if row_index == 0 else "edge times $\\tau_p$")
        for row_index, (name, ylabel) in enumerate(
            (("initial_energy", "initial energy $U(0)$"), ("plateau", "plateau $c$")), start=2
        ):
            axis = axes[row_index, column]
            for energy, color, label in (
                ("node_energy", "#7b3294", "$\\mathcal{U}_\\rho$"),
                ("edge_energy", "#008837", "$\\mathcal{U}_E$"),
            ):
                values = row_matrix(rows, energy, name)[column]
                axis.plot(config.RATIOS, values, "o-", ms=2.5, lw=1, color=color, label=label)
            axis.set_xscale("symlog", linthresh=0.25)
            if name == "plateau":
                axis.set_yscale("log")
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 3:
                axis.set_xlabel("relative rate $q/\\alpha$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    axes[2, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Inverse-progress parameterization", y=1.01)
    save_figure(fig, "f_inverse_progress_parameters")


def plot_first_step(audit: dict[str, np.ndarray]) -> None:
    """Save isolated first-step motion, rewiring and mixed-energy audit panels."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharex=True)
    q = config.ALPHA * config.RATIOS
    for index, epsilon in enumerate(config.EPSILONS):
        full = -audit["first_step_full_delta"][index, :, 1]
        rewiring = -audit["first_step_rewiring_only_delta"][index, :, 1]
        opinion = -audit["first_step_opinion_only_delta"][index, :, 1]
        axes[index].plot(q, full, "o-", ms=2.5, lw=1.1, label="full step")
        axes[index].plot(q, rewiring, "--", lw=1.2, label="rewiring only")
        axes[index].plot(q, opinion, ":", lw=1.2, label="opinion only")
        axes[index].set_xscale("symlog", linthresh=0.0025)
        axes[index].set_title(f"$\\epsilon={epsilon:g}$")
        axes[index].set_xlabel("rewiring rate $q$")
        axes[index].grid(alpha=0.2)
    axes[0].set_ylabel("first-step edge-energy drop $-\\Delta\\mathcal{U}_E$")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("The apparent initial jump is a resolved finite rewiring step", y=1.02)
    save_figure(fig, "f_first_step_audit")


def plot_final_potentials(data: dict[str, np.ndarray]) -> None:
    """Save terminal node/edge potential panels from the baseline cache."""
    x = np.linspace(-1 + 1 / config.GRID_SIZE, 1 - 1 / config.GRID_SIZE, config.GRID_SIZE)
    selected = [nearest_ratio(value) for value in (0, 1, 10, 100)]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected)))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), sharex=True)
    for column, epsilon in enumerate(config.EPSILONS):
        for row, name in enumerate(("final_node_potential", "final_edge_potential")):
            for index, color in zip(selected, colors, strict=True):
                axes[row, column].plot(
                    x,
                    data[name][column, index] / epsilon**2,
                    color=color,
                    lw=1.15,
                    label=f"$q/\\alpha={config.RATIOS[index]:g}$",
                )
            axes[row, column].grid(alpha=0.2)
            if row == 0:
                axes[row, column].set_title(f"$\\epsilon={epsilon:g}$")
            if row == 1:
                axes[row, column].set_xlabel("opinion $x$")
            if column == 0:
                axes[row, column].set_ylabel(
                    "final $\\Phi_\\rho/\\epsilon^2$" if row == 0 else "final $\\Phi_E/\\epsilon^2$"
                )
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Final interaction-potential fields returned by the solver", y=1.01)
    save_figure(fig, "f_final_potential_fields")


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save PDF/PNG copies in config.HERE at 300 dpi and close fig; the directory must exist."""
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(config.HERE / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
