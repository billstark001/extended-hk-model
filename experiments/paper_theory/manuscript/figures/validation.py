"""Validation overview, primary/sensitivity clocks and optional matched basin example."""

from __future__ import annotations

import math

import numpy as np

from experiments.paper_theory.reporting import number


def ensure_log_limits(fig):
    """Supply finite positive x limits for logarithmic panels whose empty selection has no finite data limits."""
    for axis in fig.axes:
        if axis.get_xscale() == "log" and not np.isfinite(axis.dataLim.intervalx).all():
            # An empty selection has no positive limits for the log locator.
            axis.set_xlim(0.1, 40)


def plots(out, micro, clocks, selected):
    """Render validation report figures; missing selected examples are omitted and empty log selections receive finite positive limits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#2878b5", "opinion_random": "#e07b19", "structure_random_l0": "#258445"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for i, dynamics in enumerate(("hk", "deffuant")):
        for rec in colors:
            zeta = 1 if rec == "random" else 4
            rs = sorted(
                [
                    r
                    for r in clocks
                    if r["scan"] == "main"
                    and r["dynamics"] == dynamics
                    and r["recommender"] == rec
                    and float(r["zeta"]) == zeta
                    and float(r["ratio"]) > 0
                ],
                key=lambda r: float(r["ratio"]),
            )
            x = [float(r["ratio"]) for r in rs]
            y = [number(r, "delta50") if number(r, "delta50") is not None else np.nan for r in rs]
            label = rec.replace("_random_l0", " L0").replace("_random", "") + (" z=4" if zeta == 4 else "")
            axes[i, 0].plot(x, y, "o-", color=colors[rec], label=label, ms=4)
            sensitivity = [r for r in rs if r["energy_only_added"]]
            axes[i, 0].scatter(
                [float(r["ratio"]) for r in sensitivity],
                [r["finite_delta50"] for r in sensitivity],
                facecolors="none",
                edgecolors=colors[rec],
                s=40,
            )
        axes[i, 0].set(
            xscale="log",
            xlabel=r"$q/\alpha$",
            ylabel=r"$\log(T_{50}^{E}/T_{50}^{\rho})$",
            title=dynamics.upper() + ": clocks",
        )
        axes[i, 0].axhline(0, color="0.5", lw=0.7)
        axes[i, 0].legend(fontsize=8)
        for metric, label, color in (
            ("node_w1", "Node W1", "#7a329b"),
            ("node_energy_gap", "Node energy gap", "#c43c39"),
            ("edge_energy_gap", "Edge energy gap", "#2878b5"),
        ):
            values = [
                [
                    float(r[metric])
                    for r in micro
                    if r["tier"] == "core" and r["dynamics"] == dynamics and math.isclose(float(r["ratio"]), ratio)
                ]
                for ratio in (0.25, 2, 10)
            ]
            axes[i, 1].plot(
                (0.25, 2, 10), [np.median(v) if v else np.nan for v in values], "o-", label=label, color=color
            )
        axes[i, 1].set(
            xscale="log",
            xlabel=r"$q/\alpha$",
            ylabel="Median discrepancy",
            title=dynamics.upper() + ": matched micro / pair",
        )
        axes[i, 1].legend(fontsize=8)
    fig.suptitle(
        "Filled: original joint plateau; open: post hoc energy-only sensitivity\nMissing primary points break lines; micro medians pool 5 kernels x 4 starts",
        fontsize=10,
    )
    for ext in ("png", "pdf"):
        ensure_log_limits(fig)
        fig.savefig(out / ("review_overview." + ext), dpi=180)
    plt.close(fig)
    configurations = (
        ("random", 1),
        ("opinion_random", 1),
        ("opinion_random", 4),
        ("structure_random_l0", 1),
        ("structure_random_l0", 4),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained")
    for i, dynamics in enumerate(("hk", "deffuant")):
        for rec, zeta in configurations:
            rs = sorted(
                [
                    r
                    for r in clocks
                    if r["scan"] == "main"
                    and r["dynamics"] == dynamics
                    and r["recommender"] == rec
                    and float(r["zeta"]) == zeta
                    and float(r["ratio"]) > 0
                ],
                key=lambda r: float(r["ratio"]),
            )
            label = {"random": "Random", "opinion_random": "Opinion", "structure_random_l0": "Structure L0"}[rec]
            if rec != "random":
                label += " z=" + str(zeta)
            style = "-" if zeta == 1 else "--"
            for j, key in enumerate(("rho_tau50", "edge_tau50", "delta50")):
                y = [number(r, key) if r["primary_joint_plateau"] else np.nan for r in rs]
                axes[i, j].plot([float(r["ratio"]) for r in rs], y, "o" + style, color=colors[rec], ms=3, label=label)
                sensitivity = [r for r in rs if r["energy_only_added"]]
                axes[i, j].scatter(
                    [float(r["ratio"]) for r in sensitivity],
                    [r["finite_delta50"] if key == "delta50" else number(r, key) for r in sensitivity],
                    facecolors="none",
                    edgecolors=colors[rec],
                    s=27,
                    marker="o" if zeta == 1 else "s",
                )
        for j, title in enumerate(("Node half-release", "Edge half-release", "Relative half-release")):
            axes[i, j].set(xscale="log", xlabel=r"$q/\alpha$", title=dynamics.upper() + ": " + title)
        axes[i, 0].set_ylabel(r"$\alpha T_{50}^{\rho}$")
        axes[i, 1].set_ylabel(r"$\alpha T_{50}^{E}$")
        axes[i, 2].set_ylabel(r"$\log(T_{50}^{E}/T_{50}^{\rho})$")
        axes[i, 2].axhline(0, color="0.5", lw=0.7)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(
        "epsilon=0.45, alpha=0.02; filled: joint plateau; open: post hoc energy-only tail check\nPrimary curves break at unverified design points; open circles z=1, open squares z=4",
        fontsize=10,
    )
    for ext in ("png", "pdf"):
        ensure_log_limits(fig)
        fig.savefig(out / ("clocks_with_sensitivity." + ext), dpi=180)
    plt.close(fig)
    if selected is None:
        return
    v = selected["result"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    for frames, label, style in ((v["frames"], "Micro", "-"), (v["pair"]["frames"], "Pair", "--")):
        ax[0].plot([0.02 * f["time"] for f in frames], [f["node_energy"] for f in frames], style, label=label)
        ax[1].plot(v["pair"]["axis"], frames[-1]["rho"], style, label=label)
    ax[0].set(xlabel=r"$\alpha t$", ylabel="Node interaction energy")
    ax[1].set(xlabel="Opinion", ylabel="Cell probability")
    for a in ax:
        a.legend()
    fig.suptitle("HK / Random / q/alpha=10 / initial replicate 0: different terminal states", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(out / ("basin_disagreement." + ext), dpi=180)
    plt.close(fig)
