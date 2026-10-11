"""Common-domain node/fast-edge clocks and crossover summary figures."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from experiments.paper_theory.energy_scan.expanded import config


def plot_energy_two_clock_summary(
    rows: list[dict[str, Any]], edge_fits: list[dict[str, Any]], low_parameter_rows: list[dict[str, Any]]
) -> None:
    """Render energy-curve reconstruction and node/fast-clock summary panels across the expanded scan."""

    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    figure, axes = plt.subplots(3, 3, figsize=(10.8, 9.2))
    figure.subplots_adjust(left=0.09, right=0.86, top=0.91, bottom=0.08, hspace=0.38, wspace=0.34)
    common_ratios = config.POSITIVE_RATIOS[config.POSITIVE_RATIOS <= 100.0 + 1e-12]
    color_map = plt.cm.cividis_r
    color_norm = LogNorm(vmin=float(common_ratios[0]), vmax=float(common_ratios[-1]))
    dynamics_style = {"hk": ("#2166ac", "-", "o", "HK"), "deffuant": ("#b2182b", "--", "^", "Deffuant")}
    curve_lookup = {
        (
            row["dynamics"],
            round(float(row["epsilon"]), 12),
            round(float(row["alpha"]), 12),
            round(float(row["q_over_alpha"]), 12),
            row["energy"],
        ): row
        for row in rows
    }

    def compact_log_ticks(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
        axis.set_xscale("log")
        axis.set_yscale("log")
        for dimension, values in ((axis.xaxis, x), (axis.yaxis, y)):
            positive = values[np.isfinite(values) & (values > 0)]
            if not positive.size:
                continue
            lower = float(np.min(positive))
            upper = float(np.max(positive))
            ticks = np.asarray((math.sqrt(lower * upper),)) if upper / lower < 1.01 else np.geomspace(lower, upper, 3)
            dimension.set_major_locator(FixedLocator(ticks))
            dimension.set_major_formatter(FixedFormatter([f"{value:.3g}" for value in ticks]))
            dimension.set_minor_locator(NullLocator())

    def median_parameter_path(
        subset: list[dict[str, Any]], x_name: str, y_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        ratios = sorted(
            {float(row["q_over_alpha"]) for row in subset if 0 < float(row["q_over_alpha"]) <= 100.0 + 1e-12}
        )
        grouped = [[row for row in subset if np.isclose(row["q_over_alpha"], ratio)] for ratio in ratios]
        x = np.asarray([np.median([row[x_name] for row in group]) for group in grouped])
        y = np.asarray([np.median([row[y_name] for row in group]) for group in grouped])
        return (np.asarray(ratios), x, y, grouped)

    for epsilon_index, epsilon_value in enumerate(config.EPSILONS):
        epsilon = float(epsilon_value)
        time_axis = axes[0, epsilon_index]
        for dynamics in config.DYNAMICS:
            color, line_style, _, _ = dynamics_style[dynamics]
            alpha_paths = []
            for alpha_value in config.ALPHAS:
                alpha = float(alpha_value)
                path = []
                for ratio_value in common_ratios:
                    ratio = float(ratio_value)
                    key = (dynamics, round(epsilon, 12), round(alpha, 12), round(ratio, 12))
                    edge = curve_lookup[(*key, "edge_energy")]
                    node = curve_lookup[(*key, "node_energy")]
                    path.append(math.log(edge["tau_50"] / node["tau_50"]))
                alpha_paths.append(path)
                time_axis.plot(common_ratios, path, color=color, ls=line_style, lw=0.55, alpha=0.22, zorder=1)
            median_path = np.median(np.asarray(alpha_paths), axis=0)
            time_axis.plot(common_ratios, median_path, color=color, ls=line_style, lw=1.8, zorder=2)
            for left, right, y_left, y_right in zip(
                common_ratios[:-1], common_ratios[1:], median_path[:-1], median_path[1:], strict=True
            ):
                if y_left == 0 or y_left * y_right < 0:
                    fraction = -y_left / (y_right - y_left)
                    crossing = float(np.exp(np.log(left) + fraction * (np.log(right) - np.log(left))))
                    time_axis.scatter(
                        crossing, 0.0, s=30, marker="D", facecolor="white", edgecolor=color, linewidth=1.0, zorder=4
                    )
        time_axis.axhline(0.0, color="#555555", lw=0.75, zorder=0)
        time_axis.set_xscale("log")
        time_axis.grid(alpha=0.18, which="major")
        time_axis.set_title(f"$\\epsilon={epsilon:g}$")
        time_axis.set_xlabel("relative rate $r=q/\\alpha$")
        if epsilon_index == 0:
            time_axis.set_ylabel("$\\log(T_{50}^{E}/T_{50}^{\\rho})$")
        edge_axis = axes[1, epsilon_index]
        for dynamics in config.DYNAMICS:
            color, line_style, marker, _ = dynamics_style[dynamics]
            subset = [row for row in edge_fits if row["dynamics"] == dynamics and row["epsilon"] == epsilon]
            ratios, weights, kappas, grouped = median_parameter_path(subset, "weight", "kappa")
            beta = np.asarray([np.median([row["beta"] for row in group]) for group in grouped])
            marker_areas = 20.0 + 14.0 * np.clip(beta, *config.EDGE_BETA_BOUNDS)
            edge_axis.plot(weights, kappas, color=color, ls=line_style, lw=1.1, alpha=0.85, zorder=1)
            edge_axis.scatter(
                weights,
                kappas,
                c=ratios,
                norm=color_norm,
                cmap=color_map,
                marker=marker,
                s=marker_areas,
                edgecolor=color,
                linewidth=0.45,
                zorder=2,
            )
            weak = np.asarray([not all((row["shape_identifiable"] for row in group)) for group in grouped])
            if np.any(weak):
                edge_axis.scatter(
                    weights[weak],
                    kappas[weak],
                    marker="x",
                    s=marker_areas[weak] + 12.0,
                    color="black",
                    linewidth=0.9,
                    zorder=3,
                )
        compact_log_ticks(edge_axis, weights, kappas)
        edge_axis.grid(alpha=0.18, which="major")
        edge_axis.set_xlabel("fast-layer amplitude $w$")
        if epsilon_index == 0:
            edge_axis.set_ylabel("rewiring coefficient $\\kappa$")
        node_axis = axes[2, epsilon_index]
        for dynamics in config.DYNAMICS:
            color, line_style, marker, _ = dynamics_style[dynamics]
            subset = [
                row
                for row in low_parameter_rows
                if row["dynamics"] == dynamics
                and row["epsilon"] == epsilon
                and (row["energy"] == "node_energy")
                and (row["model"] == "gompertz_hazard_2")
            ]
            ratios, hazards, accelerations, grouped = median_parameter_path(subset, "initial_hazard", "acceleration")
            node_axis.plot(hazards, accelerations, color=color, ls=line_style, lw=1.1, alpha=0.85, zorder=1)
            node_axis.scatter(
                hazards,
                accelerations,
                c=ratios,
                norm=color_norm,
                cmap=color_map,
                marker=marker,
                s=27,
                edgecolor=color,
                linewidth=0.45,
                zorder=2,
            )
            poor = np.asarray(
                [np.median([row["maximum_absolute_relative_error"] for row in group]) > 0.2 for group in grouped]
            )
            if np.any(poor):
                node_axis.scatter(
                    hazards[poor],
                    accelerations[poor],
                    marker="s",
                    s=58,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.85,
                    zorder=3,
                )
        compact_log_ticks(node_axis, hazards, accelerations)
        node_axis.grid(alpha=0.18, which="major")
        node_axis.set_xlabel("node initial hazard $h_0$")
        if epsilon_index == 0:
            node_axis.set_ylabel("node acceleration $g$")
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.04,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    dynamics_handles = [
        plt.Line2D((), (), color=color, ls=line_style, marker=marker, ms=5, label=label)
        for color, line_style, marker, label in dynamics_style.values()
    ]
    diagnostic_handles = [
        plt.Line2D((), (), marker="x", color="black", ls="none", ms=5, label="weak edge-shape identification"),
        plt.Line2D((), (), marker="s", mfc="none", mec="black", ls="none", ms=5, label="poor node descriptor"),
    ]
    figure.legend(
        handles=dynamics_handles + diagnostic_handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.985),
        ncol=4,
        fontsize=7.5,
        frameon=False,
    )
    color_axis = figure.add_axes((0.895, 0.29, 0.018, 0.52))
    color_bar = figure.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=color_map), cax=color_axis)
    color_bar.set_label("path parameter $q/\\alpha$")
    beta_handles = [
        plt.Line2D(
            (),
            (),
            marker="o",
            ms=math.sqrt(20.0 + 14.0 * beta),
            mfc="#777777",
            mec="#202020",
            mew=0.25,
            ls="none",
            label=f"{beta:g}",
        )
        for beta in (0.5, 1.0, 2.0, 5.0)
    ]
    figure.legend(
        handles=beta_handles,
        title="edge marker area: $\\beta$",
        loc="lower left",
        bbox_to_anchor=(0.87, 0.075),
        fontsize=7.0,
        title_fontsize=7.0,
        frameon=False,
        labelspacing=0.55,
        handletextpad=0.5,
    )
    save_figure(figure, "f_energy_two_clock_summary")
