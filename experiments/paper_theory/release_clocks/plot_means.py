"""Observed finite-window versus empirical-fit mean-release comparisons."""

import matplotlib.pyplot as plt

from common.plotting.figures import save
from experiments.paper_theory.data import read_csv


def plot_existing(root):
    """Render observed versus empirical-fit integral means from saved release-clock tables."""
    rows = read_csv(root / "tables" / "mean_release.csv")
    rules = sorted({row["dynamics"] for row in rows})
    fig, axes = plt.subplots(2, len(rules), figsize=(5 * len(rules), 7), squeeze=False, sharex=True)
    colors = {
        e: plt.get_cmap("viridis")(v) for e, v in zip(sorted({float(r["epsilon"]) for r in rows}), (0.15, 0.5, 0.85))
    }
    eligible = [
        r
        for r in rows
        if r["converged"] == "True"
        and r["fit_status"] == "available"
        and r["fit_low_quality"] == "False"
        and float(r["node_fit_max_quantile_error"]) <= 0.1
        and float(r["edge_tail_fraction"]) <= 0.05
        and float(r["r"]) > 0
    ]
    for column, rule in enumerate(rules):
        for epsilon, color in colors.items():
            subset = [r for r in eligible if r["dynamics"] == rule and float(r["epsilon"]) == epsilon]
            for alpha in sorted({float(r["alpha"]) for r in subset}):
                group = sorted((r for r in subset if float(r["alpha"]) == alpha), key=lambda r: float(r["r"]))
                if not group:
                    continue
                x = [float(r["r"]) for r in group]
                label = f"ε={epsilon:g}" if alpha == min(float(r["alpha"]) for r in subset) else None
                axes[0, column].plot(
                    x,
                    [float(r["fitted_edge_to_node"]) for r in group],
                    color=color,
                    alpha=0.65,
                    linewidth=1,
                    marker=".",
                    markersize=3,
                    label=label,
                )
                axes[1, column].plot(
                    x,
                    [float(r["observed_edge_to_node"]) for r in group],
                    color=color,
                    alpha=0.65,
                    linewidth=1,
                    marker=".",
                    markersize=3,
                )
        axes[0, column].set_title(rule.upper())
        axes[0, column].legend(frameon=False)
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.axhline(1, color=".5", linewidth=0.7, linestyle="--")
            axis.grid(alpha=0.15)
        axes[1, column].set_xlabel(r"$r=q/\alpha$")
    axes[0, 0].set_ylabel("Fitted mean edge / node release time")
    axes[1, 0].set_ylabel("Finite-window edge / node area")
    fig.tight_layout()
    save(fig, root / "figures" / "mean_release_parameters")
    (root / "figures" / "mean_release_parameters.caption.txt").write_text(
        f"Whole-curve release clocks in tau units. {len(eligible)} of {len(rows)} trajectories shown: "
        "positive r, converged, available fit, no low-quality edge flag, node fit maximum quantile error <=0.1, "
        "fitted edge tail fraction <=0.05. All rows remain in the CSV. Upper: infinite empirical-fit integral; "
        "lower: observed nonnegative remaining-energy area with tail-median plateau. Rebounds are retained. "
        "Different alpha values are overlaid, not treated as independent replicates.\n"
    )
