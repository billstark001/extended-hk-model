"""First-step scaling, scaled-time collapse and uniform-law derivative figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper_theory.energy_scan.scaling import config


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save PDF/PNG copies in config.HERE at 300 dpi and close fig; the directory must exist."""
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(config.HERE / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_channel_tests(data: dict[str, np.ndarray]) -> None:
    """Save isolated alpha/q first-step scaling panels and their continuum coefficient comparisons."""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    alpha = data["alpha_probe"]
    alpha_coefficient = float(data["alpha_coefficient_continuum"])
    axes[0].loglog(alpha, -data["alpha_delta_u"], "o-", ms=3.5, label="Go measure step")
    axes[0].loglog(alpha, alpha_coefficient * alpha, "--", label="$A_0\\alpha$")
    axes[0].set_xlabel("influence $\\alpha$")
    axes[0].set_ylabel("one-step dissipation $-\\Delta\\mathcal{U}_\\rho$")
    axes[0].set_title("Opinion-channel scaling")
    axes[0].legend(frameon=False, fontsize=8)
    q = data["q_probe"]
    q_coefficient = float(data["q_coefficient_continuum"])
    axes[1].loglog(q, -data["q_delta_ue"], "o-", ms=3.5, label="Go rewiring step")
    axes[1].loglog(q, q_coefficient * q, "--", label="$R_0q$")
    axes[1].set_xlabel("rewiring rate $q$")
    axes[1].set_ylabel("one-step dissipation $-\\Delta\\mathcal{U}_E$")
    axes[1].set_title("Rewiring-channel scaling")
    axes[1].legend(frameon=False, fontsize=8)
    epsilon = data["epsilon_probe"]
    axes[2].plot(epsilon, data["epsilon_analytic_u"], label="analytic $\\mathcal{U}_{\\rho,0}$")
    axes[2].plot(epsilon, data["epsilon_analytic_ue"], label="analytic $\\mathcal{U}_{E,0}$")
    axes[2].plot(epsilon, data["epsilon_numerical_u"], ".", ms=2.2, label="$B=161$ $\\mathcal{U}_{\\rho,0}$")
    axes[2].plot(epsilon, data["epsilon_numerical_ue"], ".", ms=2.2, label="$B=161$ $\\mathcal{U}_{E,0}$")
    axes[2].set_xlabel("confidence radius $\\epsilon$")
    axes[2].set_ylabel("uniform-start energy")
    axes[2].set_title("Explicit $\\epsilon$ dependence")
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Frozen-state parameter relations", y=1.03, fontsize=12)
    save_figure(fig, "f_frozen_channel_scaling")


def plot_dynamic_collapse(data: dict[str, np.ndarray]) -> None:
    """Save energy/topology curves in scaled time across alpha at fixed q/alpha."""
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, config.ALPHA_COLLAPSE.size))
    series = (
        (data["energy_rho"], "$\\mathcal{U}_\\rho[\\rho]$"),
        (data["energy_edge"], "$\\mathcal{U}_E[E]$"),
        (data["topology_excess"], "$\\Delta\\mathcal{U}_{\\rm top}=\\mathcal{U}_E-2\\mathcal{U}_\\rho$"),
    )
    for column, gamma in enumerate(data["gamma_collapse"]):
        axes[0, column].set_title(f"$q/\\alpha={gamma:g}$")
        for row, (values, ylabel) in enumerate(series):
            for alpha_index, (alpha, color) in enumerate(zip(config.ALPHA_COLLAPSE, colors, strict=True)):
                axes[row, column].plot(
                    data["tau"], values[column, alpha_index], color=color, lw=1.25, label=f"$\\alpha={alpha:g}$"
                )
            axes[row, column].grid(alpha=0.2)
            if column == 0:
                axes[row, column].set_ylabel(ylabel)
            if row == 2:
                axes[row, column].set_xlabel("scaled time $\\tau=\\alpha t$")
    for row, (values, _) in enumerate(series):
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max()))
        padding = 0.06 * max(high - low, 1e-10)
        for column in range(3):
            axes[row, column].set_ylim(low - padding, high + padding)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Scaled-time collapse at fixed $\\gamma=q/\\alpha$", y=1.01, fontsize=12)
    save_figure(fig, "f_dynamic_ratio_collapse")


def plot_field_collapse(data: dict[str, np.ndarray]) -> None:
    """Save terminal potential-field collapse panels at fixed q/alpha."""
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, config.ALPHA_COLLAPSE.size))
    x = data["x"]
    phi_rho = data["phi_rho"][:, :, -1]
    phi_edge = data["phi_edge"][:, :, -1]
    fields = (
        (phi_rho, "$\\Phi_\\rho(x,\\tau=10)$"),
        (phi_edge, "$\\Phi_E(x,\\tau=10)$"),
        (phi_edge - phi_rho, "$\\Delta\\Phi=\\Phi_E-\\Phi_\\rho$"),
    )
    for column, gamma in enumerate(data["gamma_collapse"]):
        axes[0, column].set_title(f"$q/\\alpha={gamma:g}$")
        for row, (values, ylabel) in enumerate(fields):
            for alpha_index, (alpha, color) in enumerate(zip(config.ALPHA_COLLAPSE, colors, strict=True)):
                axes[row, column].plot(
                    x, values[column, alpha_index], color=color, lw=1.2, label=f"$\\alpha={alpha:g}$"
                )
            axes[row, column].grid(alpha=0.2)
            if column == 0:
                axes[row, column].set_ylabel(ylabel)
            if row == 2:
                axes[row, column].set_xlabel("opinion $x$")
    for row, (values, _) in enumerate(fields):
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max()))
        padding = 0.06 * max(high - low, 1e-10)
        for column in range(3):
            axes[row, column].set_ylim(low - padding, high + padding)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Interaction-energy landscape collapse at fixed $q/\\alpha$", y=1.01, fontsize=12)
    save_figure(fig, "f_potential_field_ratio_collapse")


def plot_epsilon_derivatives(data: dict[str, np.ndarray]) -> None:
    """Save uniform-law energy and confidence-derivative comparisons with analytic coefficients."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    epsilon = data["epsilon_probe"]
    axes[0].plot(epsilon, data["epsilon_analytic_du"], label="analytic")
    axes[0].plot(epsilon, data["epsilon_numerical_du"], ".", ms=2.4, label="$B=161$")
    axes[0].set_ylabel("$\\partial_\\epsilon\\mathcal{U}_{\\rho,0}$")
    axes[0].set_title("Node energy")
    axes[1].plot(epsilon, data["epsilon_analytic_due"], label="analytic")
    axes[1].plot(epsilon, data["epsilon_numerical_due"], ".", ms=2.4, label="$B=161$")
    axes[1].set_ylabel("$\\partial_\\epsilon\\mathcal{U}_{E,0}$")
    axes[1].set_title("Edge energy")
    for axis in axes:
        axis.set_xlabel("confidence radius $\\epsilon$")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Discordant-mass derivative identities", y=1.04, fontsize=12)
    save_figure(fig, "f_epsilon_derivative_identity")
