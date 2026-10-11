"""Edge-mixture parameter paths with quality and identifiability indicators."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from experiments.paper_theory.energy_scan.expanded import config


def plot_edge_timescale_parameters(edge_fits: list[dict[str, Any]], low_parameter_rows: list[dict[str, Any]]) -> None:
    """Render edge-mixture parameter trends with identifiability and fit-quality markers."""

    from experiments.paper_theory.energy_scan.expanded.plot.core import save_figure

    figure, axes = plt.subplots(4, 3, figsize=(11.0, 11.2))
    figure.subplots_adjust(left=0.1, right=0.86, top=0.9, bottom=0.07, hspace=0.3, wspace=0.34)
    color_map = plt.cm.cividis_r
    color_norm = LogNorm(vmin=float(config.POSITIVE_RATIOS[0]), vmax=float(config.POSITIVE_RATIOS[-1]))
    convergence = {
        (row["dynamics"], row["epsilon"], round(float(row["alpha"]), 12), round(float(row["q_over_alpha"]), 12)): bool(
            row["trajectory_converged"]
        )
        for row in edge_fits
    }

    def draw_path(
        axis: plt.Axes,
        subset: list[dict[str, Any]],
        x_name: str,
        y_name: str,
        *,
        poor_name: str,
        poor_threshold: float,
        mark_weak_shape: bool,
        size_name: str | None = None,
    ) -> None:
        for alpha in config.ALPHAS:
            alpha_rows = sorted(
                (row for row in subset if np.isclose(row["alpha"], alpha) and row["q_over_alpha"] > 0),
                key=lambda row: row["q_over_alpha"],
            )
            axis.plot(
                [row[x_name] for row in alpha_rows],
                [row[y_name] for row in alpha_rows],
                color="#a9a9a9",
                lw=0.55,
                alpha=0.45,
                zorder=1,
            )
        ratios = sorted({float(row["q_over_alpha"]) for row in subset if row["q_over_alpha"] > 0})
        grouped = [[row for row in subset if np.isclose(row["q_over_alpha"], ratio)] for ratio in ratios]
        x = np.asarray([np.median([row[x_name] for row in group]) for group in grouped])
        y = np.asarray([np.median([row[y_name] for row in group]) for group in grouped])
        ratio_array = np.asarray(ratios)
        if size_name is None:
            marker_areas = np.full_like(ratio_array, 24.0)
        else:
            size_values = np.asarray([np.median([row[size_name] for row in group]) for group in grouped])
            marker_areas = 20.0 + 14.0 * np.clip(size_values, *config.EDGE_BETA_BOUNDS)
        axis.plot(x, y, color="#333333", lw=1.0, zorder=2)
        axis.scatter(
            x,
            y,
            c=ratio_array,
            norm=color_norm,
            cmap=color_map,
            s=marker_areas,
            edgecolors="#202020",
            linewidths=0.25,
            zorder=3,
        )
        poor = np.asarray([np.median([row[poor_name] for row in group]) > poor_threshold for group in grouped])
        if np.any(poor):
            axis.scatter(
                x[poor],
                y[poor],
                marker="s",
                s=marker_areas[poor] + 24.0,
                facecolors="none",
                edgecolors="black",
                linewidth=0.8,
                zorder=4,
            )
        if mark_weak_shape:
            weak = np.asarray([not all((row["shape_identifiable"] for row in group)) for group in grouped])
            if np.any(weak):
                axis.scatter(
                    x[weak], y[weak], marker="x", s=marker_areas[weak] + 10.0, color="black", linewidth=0.9, zorder=5
                )
        censored = []
        for group in grouped:
            row = group[0]
            censored.append(
                not all(
                    (
                        convergence.get(
                            (
                                row["dynamics"],
                                row["epsilon"],
                                round(float(item["alpha"]), 12),
                                round(float(item["q_over_alpha"]), 12),
                            ),
                            True,
                        )
                        for item in group
                    )
                )
            )
        censored = np.asarray(censored)
        if np.any(censored):
            axis.scatter(
                x[censored],
                y[censored],
                marker="o",
                s=marker_areas[censored] + 24.0,
                facecolors="none",
                edgecolors="black",
                linewidth=0.9,
                zorder=6,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        for axis_dimension, values in ((axis.xaxis, x), (axis.yaxis, y)):
            positive = values[np.isfinite(values) & (values > 0)]
            if positive.size:
                lower = float(np.min(positive))
                upper = float(np.max(positive))
                if upper / lower < 1.01:
                    ticks = np.asarray((math.sqrt(lower * upper),))
                else:
                    ticks = np.geomspace(lower, upper, 3)
                axis_dimension.set_major_locator(FixedLocator(ticks))
                axis_dimension.set_major_formatter(FixedFormatter([f"{value:.3g}" for value in ticks]))
                axis_dimension.set_minor_locator(NullLocator())
        axis.grid(alpha=0.2, which="major")

    for dynamics_index, dynamics in enumerate(config.DYNAMICS):
        edge_row = 2 * dynamics_index
        node_row = edge_row + 1
        for epsilon_index, epsilon in enumerate(config.EPSILONS):
            edge_subset = [row for row in edge_fits if row["dynamics"] == dynamics and row["epsilon"] == epsilon]
            node_subset = [
                row
                for row in low_parameter_rows
                if row["dynamics"] == dynamics
                and row["epsilon"] == epsilon
                and (row["energy"] == "node_energy")
                and (row["model"] == "gompertz_hazard_2")
            ]
            edge_axis = axes[edge_row, epsilon_index]
            node_axis = axes[node_row, epsilon_index]
            draw_path(
                edge_axis,
                edge_subset,
                "weight",
                "kappa",
                poor_name="rms_normalized_energy_error",
                poor_threshold=0.03,
                mark_weak_shape=True,
                size_name="beta",
            )
            draw_path(
                node_axis,
                node_subset,
                "initial_hazard",
                "acceleration",
                poor_name="maximum_absolute_relative_error",
                poor_threshold=0.2,
                mark_weak_shape=False,
            )
            if edge_row == 0:
                edge_axis.set_title(f"$\\epsilon={epsilon:g}$")
            edge_axis.set_xlabel("fast fraction $w$")
            node_axis.set_xlabel("initial hazard $h_0$")
            if epsilon_index == 0:
                edge_axis.set_ylabel(f"{dynamics.upper()} edge: $\\kappa$", labelpad=7)
                node_axis.set_ylabel(f"{dynamics.upper()} node: $g$", labelpad=7)
    color_axis = figure.add_axes((0.902, 0.27, 0.018, 0.53))
    color_bar = figure.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=color_map), cax=color_axis)
    color_bar.set_label("path parameter $q/\\alpha$")
    legend_handles = [
        plt.Line2D((), (), color="#a9a9a9", lw=0.8, label="individual $\\alpha$"),
        plt.Line2D((), (), color="#333333", lw=1.2, label="median path"),
        plt.Line2D((), (), marker="s", ms=6, mfc="none", mec="black", ls="none", label="poor fit"),
        plt.Line2D((), (), marker="x", ms=6, color="black", ls="none", label="weak shape identification"),
        plt.Line2D((), (), marker="o", ms=6, mfc="none", mec="black", ls="none", label="right-censored at $\\tau=120$"),
    ]
    figure.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=7.5, frameon=False
    )
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
        bbox_to_anchor=(0.875, 0.055),
        fontsize=7.0,
        title_fontsize=7.0,
        frameon=False,
        labelspacing=0.55,
        handletextpad=0.5,
    )
    figure.suptitle("$q/\\alpha$-parametrized paths in physical parameter planes", y=0.995)
    save_figure(figure, "f_edge_two_timescale_parameters")
