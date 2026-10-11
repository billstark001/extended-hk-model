"""Operator discrepancy and fixed-coefficient numerical-convergence panels."""

from __future__ import annotations

from common.plotting.batch import plotting, savefig


def plot_operator(rows, mesh, out):
    """Render measure/Fokker-Planck discrepancy and numerical-convergence panels from analysis tables."""
    if not rows and not mesh:
        return []
    plt = plotting()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for d in ("hk", "deffuant"):
        group = [r for r in rows if r["dynamics"] == d]
        for j, metric in enumerate(("node_hellinger", "edge_hellinger", "edge_energy_gap")):
            axs[j].scatter([r["alpha"] for r in group], [r[metric] for r in group], label=d, s=10)
            axs[j].set_xlabel("influence")
            axs[j].set_ylabel(metric)
            axs[j].legend()
    savefig(fig, out / "figures", "operator_discrepancies")
    if mesh:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))
        for i, kind in enumerate(("numerical_step", "spatial_grid")):
            group = [r for r in mesh if r["comparison"] == kind]
            for d in ("hk", "deffuant"):
                g = [r for r in group if r["dynamics"] == d]
                if g:
                    axs[i].scatter(
                        [r["h"] if i == 0 else 2 / r["b"] for r in g], [r["edge_hellinger"] for r in g], s=12, label=d
                    )
            axs[i].set_xlabel("numerical h" if i == 0 else "cell width")
            axs[i].set_ylabel("edge Hellinger to adjacent refinement")
            if axs[i].get_legend_handles_labels()[0]:
                axs[i].legend()
        savefig(fig, out / "figures", "fixed_coefficient_convergence")
    return ["operator_discrepancies.pdf"] + (["fixed_coefficient_convergence.pdf"] if mesh else [])
