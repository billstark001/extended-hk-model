"""Paired-clock, budget and sampled crossing panels from validation records."""

from __future__ import annotations

import numpy as np

from common.io.json import atomic_json
from common.plotting.batch import plotting, savefig


def plot_clocks(items, rows, out):
    """Render direct paired-clock, budget and crossing panels from validated analysis records."""
    if not rows:
        return []
    plt = plotting()
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for i, d in enumerate(("hk", "deffuant")):
        for rec, z in (
            ("random", 1),
            ("opinion_random", 1),
            ("opinion_random", 4),
            ("structure_random_l0", 1),
            ("structure_random_l0", 4),
        ):
            group = sorted(
                [
                    r
                    for r in rows
                    if r["dynamics"] == d
                    and r["recommender"] == rec
                    and r["zeta"] == z
                    and r.get("scan") == "main"
                    and r["ratio"] > 0
                ],
                key=lambda r: r["ratio"],
            )
            for j, metric in enumerate(("rho_T50", "edge_T50", "delta50")):
                valid = [r for r in group if r[metric] is not None and r["rho_plateau_verified"]]
                if valid:
                    axs[i, j].plot([r["ratio"] for r in valid], [r[metric] for r in valid], "o-", label=f"{rec}, z={z}")
            axs[i, 2].axhline(0, color="gray", linewidth=0.6)
    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set_xlabel(r"$q/\alpha$")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    for j, name in enumerate(("node T50", "edge T50", "ln(edge T50 / node T50)")):
        for i in range(2):
            axs[i, j].set_ylabel(name)
    savefig(fig, out / "figures", "direct_relative_clocks")
    # One representative shows mass, actual drift and both conditional landscapes.
    r = min(items, key=lambda r: (abs(r["job"]["meta"]["ratio"] - 2), r["job"]["id"]))
    snapshots = [f for f in r["result"]["frames"] if "rho" in f]
    f = min(snapshots, key=lambda f: abs(r["job"]["meta"]["alpha"] * f["time"] - 2)) if snapshots else None
    if f:
        x = np.array(r["result"]["axis"])
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].plot(x, f["rho"])
        axs[0].set_ylabel("node cell mass")
        axs[1].plot(x, f["velocity"])
        axs[1].axhline(0, color="gray", linewidth=0.6)
        axs[1].set_ylabel("actual velocity")
        mask = np.asarray(f["rho"]) > 1e-6
        ep = np.array([np.nan if a is None else a for a in f["edge_potential"]])
        axs[2].plot(x, f["node_potential"], label="population")
        axs[2].plot(x[mask], ep[mask], label="edge, supported rows")
        axs[2].legend()
        axs[2].set_ylabel("interaction landscape")
        savefig(fig, out / "figures", "supported_landscape")
    frames = r["result"]["frames"]
    tau = [r["job"]["meta"]["alpha"] * f["time"] for f in frames]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for name in ("node_energy", "edge_energy", "reference_energy"):
        axs[0].plot(tau, [f[name] for f in frames], label=name)
    for name in ("budget_rewire", "budget_motion"):
        axs[1].plot(tau, [f[name] for f in frames], label=name)
    axs[1].plot(tau, [f["edge_energy"] - frames[0]["edge_energy"] for f in frames], "--", label="total edge change")
    for ax in axs:
        ax.set_xlabel(r"$\alpha t$")
        ax.legend(fontsize=7)
    savefig(fig, out / "figures", "channel_energy_budget")
    atomic_json(
        out / "figure_case.json",
        {"id": r["job"]["id"], "meta": r["job"]["meta"], "landscape_step": f["step"] if f else None},
    )
    return ["direct_relative_clocks.pdf", "supported_landscape.pdf", "channel_energy_budget.pdf"]
