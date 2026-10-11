"""Figures for conditional ensembles, full production controls and long clock traces."""

from __future__ import annotations

import numpy as np

from common.io.json import read_result
from common.plotting.batch import plotting, savefig
from experiments.paper_theory.manuscript.reports.tables import number
from experiments.paper_theory.manuscript.reports.tables import read_table as read_csv


def plots(micro, clock, root, out, estimates, pop, ensemble_controls):
    """Render manuscript ensemble/clock/population panels from source-validated reduced records and their review tables."""
    plt = plotting()
    directory = out / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    grid = read_csv(root / "analysis" / "micro" / "trajectory_errors.csv")
    for initial in range(4):
        for quad, style in ((9, "--"), (33, "-")):
            rows = sorted(
                [
                    r
                    for r in grid
                    if r["role"] == "basin_grid"
                    and int(r["initial_replicate"]) == initial
                    and int(r["quadrature"]) == quad
                ],
                key=lambda r: int(r["b"]),
            )
            for axis, metric in zip(axes, ("node_w1", "node_energy_gap")):
                axis.plot(
                    [int(r["b"]) for r in rows],
                    [number(r, metric) for r in rows],
                    style,
                    marker="o",
                    color=f"C{initial}",
                    label=f"Initial {initial}, Q={quad}",
                )
    axes[0].set(xlabel="Opinion bins B", ylabel="Final node W1")
    axes[1].set(xlabel="Opinion bins B", ylabel="Normalized node energy trajectory gap")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("HK Random, q/alpha=10, N=500; unchanged micro process; tau=30")
    savefig(fig, directory, "basin_grid_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for initial in range(4):
        for quad, style in ((9, "--"), (33, "-")):
            rs = sorted(
                [r for r in ensemble_controls if r["initial"] == initial and r["quadrature"] == quad],
                key=lambda r: r["pair_bins"],
            )
            for ax, metric, low, high in (
                (axes[0], "mean_rho_w1", "w1_lower", "w1_upper"),
                (axes[1], "node_gap_of_mean_curve", "node_gap_lower", "node_gap_upper"),
            ):
                x = [r["pair_bins"] for r in rs]
                ax.plot(
                    x,
                    [r[metric] for r in rs],
                    style,
                    marker="o",
                    color=f"C{initial}",
                    label=f"Initial {initial}, Q={quad}",
                )
                if quad == 33:
                    ax.fill_between(x, [r[low] for r in rs], [r[high] for r in rs], color=f"C{initial}", alpha=0.12)
    axes[0].set(xlabel="Pair opinion bins B", ylabel="Mean-distribution W1 (common 81 bins)")
    axes[1].set(xlabel="Pair opinion bins B", ylabel="Normalized node gap of mean energy curve")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Same 16 micro processes per initial; conditional bootstrap bands for Q=33")
    savefig(fig, directory, "conditional_ensemble_grid")

    distributions = read_result(root / "analysis" / "micro" / "ensemble_distributions.json.gz")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for initial, ax in enumerate(axes.flat):
        r = next(
            r
            for r in distributions
            if r["dynamics"] == "hk"
            and r["recommender"] == "random"
            and r["ratio"] == 10
            and r["n"] == 500
            and r["b"] == 161
            and r["quadrature"] == 33
            and r["initial_replicate"] == initial
        )
        axis = np.asarray(r["axis"])
        ax.step(
            axis, np.asarray(r["micro_mean_rho"]) * 161 / 2, where="mid", label="Micro mean, 16 processes", color="C0"
        )
        ax.step(
            axis, np.asarray(r["pair_rho"]) * 161 / 2, where="mid", label="Pair prediction", color="C3", linestyle="--"
        )
        ax.set(
            xlabel="Opinion",
            ylabel="Density",
            title=f"Initial {initial}: W1={r['node_w1']:.3f} [{r['node_w1_lower']:.3f}, {r['node_w1_upper']:.3f}]",
        )
        ax.legend(fontsize=7)
    fig.suptitle("Conditional finite-window mean distributions; N=500, B=161, Q=33, tau=30")
    savefig(fig, directory, "conditional_terminal_distributions")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for n, ax in zip((500, 1000, 2000), axes):
        rs = [r for r in pop if r["meta"]["n"] == n and r["meta"]["initial_replicate"] == 0]
        times = 0.02 * rs[0]["times"]
        for r in rs:
            ax.plot(times, r["node"], color="gray", alpha=0.45, lw=0.65)
        ax.plot(times, np.asarray([r["node"] for r in rs]).mean(axis=0), label="Micro mean", color="C0")
        ax.plot(times, rs[0]["node_pair"], "--", label="Pair", color="C3")
        ax.set(xlabel=r"$\alpha t$", ylabel="Node energy", title=f"N={n}, four processes")
        ax.legend(fontsize=8)
    fig.suptitle("Fixed numerical setting B=161, Q=33; seed index 0; different-size initial graphs")
    savefig(fig, directory, "population_transient_energy")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    labels = {"random": "Random", "opinion_random": "Opinion4", "structure_random_l0": "Structure L0"}
    colors = {"random": "C0", "opinion_random": "C1", "structure_random_l0": "C2"}
    for i, d in enumerate(("hk", "deffuant")):
        for j, channel in enumerate(("node", "edge")):
            ax = axes[i, j]
            rs = [r for r in estimates if r["dynamics"] == d and r["channel"] == channel]
            rs.sort(key=lambda r: (r["initial"], r["recommender"]))
            for y, r in enumerate(rs):
                ax.errorbar(
                    r["micro_tau50"],
                    y,
                    xerr=[[r["micro_tau50"] - r["lower"]], [r["upper"] - r["micro_tau50"]]],
                    fmt="o",
                    color=colors[r["recommender"]],
                    capsize=3,
                )
                ax.plot(r["pair_tau50"], y, "x", color="black", ms=7)
            ax.set_yticks(range(len(rs)), [f"{labels[r['recommender']]} / init {r['initial']}" for r in rs], fontsize=8)
            ax.set(xlabel=r"$\alpha T_{50}$ of mean energy curve", title=d.upper() + ": " + channel)
    fig.suptitle("Micro means: conditional 95% bootstrap intervals (8 processes); black x: pair")
    savefig(fig, directory, "conditional_clock_predictions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for z, ax in zip((1, 4), axes):
        for d, style in (("hk", "-"), ("deffuant", "--")):
            rs = sorted(
                [
                    r["row"]
                    for r in clock
                    if r["meta"]["role"] == "crossing"
                    and r["meta"]["dynamics"] == d
                    and r["meta"]["zeta"] == z
                    and r["meta"]["b"] == 161
                    and r["meta"]["quadrature"] == 33
                ],
                key=lambda r: r["ratio"],
            )
            for p, color in ((10, "C0"), (50, "C1"), (90, "C2")):
                ax.plot(
                    [r["ratio"] for r in rs],
                    [r[f"delta{p}"] if r["energy_confirmed"] else np.nan for r in rs],
                    style,
                    color=color,
                    marker="o" if d == "hk" else "s",
                    ms=3,
                    label=f"{d.upper()} {p}%",
                )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(xlabel=r"$q/\alpha$", ylabel=r"$\log(T_{edge}/T_{node})$", title=f"Opinion zeta={z}")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Energy-confirmed mesoscopic clocks; B=161, Q=33; progress-specific order")
    savefig(fig, directory, "progress_dependent_crossings")

    special = next(
        r
        for r in clock
        if r["meta"]["role"] == "extension"
        and r["meta"]["dynamics"] == "hk"
        and r["meta"]["recommender"] == "opinion_random"
        and r["meta"]["zeta"] == 4
        and r["meta"]["ratio"] == 4
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.7), layout="constrained")
    for ch, label, color in (("node", "Node", "C0"), ("edge", "Edge", "C1")):
        ax.plot(0.02 * special["times"], special[ch] / special[ch][0], label=label, color=color)
    ax.axvline(120, color="gray", linestyle="--", label="Old observation limit")
    ax.set(
        xlabel=r"$\alpha t$",
        ylabel="Energy / initial energy",
        title="HK Opinion4, q/alpha=4: late node relaxation; joint stop at tau=160",
    )
    ax.legend(fontsize=8)
    savefig(fig, directory, "extended_late_relaxation")
