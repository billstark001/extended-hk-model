"""Focused figures for parameter-crossing and pathway reviews."""

import matplotlib as mpl
import numpy as np

from common.plotting import plt_figure, plt_save_and_close

from ..recommendation_shared.plot import label, measurement_label
from . import design as cfg


def plot_bridge_replicates(runs, output, settings):
    """Show every paired effect so sparse, influential cases remain visible."""
    selected = [(runs["influence"] == influence) & (runs["rewiring"] == rewiring)
                for influence, rewiring in settings]
    late = runs[(runs["stage"] == "late") & np.logical_or.reduce(selected)]
    parameters = measurement_label(late)
    reposts = sorted(late["repost"].unique())
    for metric, stem, ylabel in (
        ("separation_change", "bridge_replicates", "Bridge − switch: opinion separation"),
        ("excess_change", "bridge_excess_replicates", "Bridge − switch: excess concordance"),
    ):
        fig, axes = plt_figure(n_row=(len(settings) + 1) // 2, n_col=2, hw_ratio=0.65, sharey=True)
        fig.suptitle(f"Late bridge intervention: individual paired effects over 512 steps\n{parameters}")
        flat = np.asarray(axes, dtype=object).reshape(-1)
        for ax, (influence, rewiring) in zip(flat, settings):
            cell = late[(late["influence"] == influence) & (late["rewiring"] == rewiring)]
            ax.axhline(0, color="0.6", lw=1)
            for index, repost in enumerate(reposts):
                paired = cell[cell["repost"] == repost].pivot(index="task_id", columns="arm", values=metric)
                if not {"bridge", "switch"} <= set(paired.columns):
                    continue
                differences = (paired["bridge"] - paired["switch"]).dropna()
                if differences.empty:
                    continue
                offset = np.linspace(-0.1, 0.1, len(differences)) if len(differences) > 1 else [0]
                ax.scatter(index + np.asarray(offset), differences, s=24, color="tab:blue")
                ax.plot(index, differences.mean(), "_", color="black", ms=14, mew=2)
                ax.text(index, 0.97, f"n={len(differences)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8)
            ax.set(title=f"α={influence:g}, q={rewiring:g}", ylabel=ylabel,
                   xticks=range(len(reposts)), xticklabels=[f"p={p:g}" for p in reposts])
        for ax in flat[len(settings):]:
            ax.set_visible(False)
        plt_save_and_close(fig, str(output / stem))


def plot_reference_pbs_grid(counts, output):
    """Display PbS counts and denominators without hiding unplanned cells."""
    parameters = measurement_label(counts)
    if counts.duplicated(["alpha", "q", "p"]).any():
        raise ValueError("a pathway grid requires one count per parameter cell")
    if not ((counts["n"] > 0) & (counts["PbS"] >= 0) & (counts["PbS"] <= counts["n"])).all():
        raise ValueError("pathway counts must satisfy 0 <= PbS <= n with n > 0")
    influences, rewirings = sorted(counts["alpha"].unique()), sorted(counts["q"].unique())
    reposts = sorted(counts["p"].unique())
    fig, axes = plt_figure(n_col=len(reposts), hw_ratio=0.9, total_width=10)
    flat = np.asarray(axes, dtype=object).reshape(-1)
    fig.suptitle(f"Reference pathways: PbS / completed repeats (16-step observations)\n{parameters}")
    for ax, repost in zip(flat, reposts):
        matrix = np.full((len(influences), len(rewirings)), np.nan)
        labels = {}
        for row in counts[counts["p"] == repost].itertuples():
            coordinate = influences.index(row.alpha), rewirings.index(row.q)
            matrix[coordinate] = row.PbS / row.n
            labels[coordinate] = f"{row.PbS}/{row.n}"
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1,
                          cmap=mpl.colormaps["Blues"].with_extremes(bad="0.9"), aspect="auto")
        for i in range(len(influences)):
            for j in range(len(rewirings)):
                ax.text(j, i, labels.get((i, j), "—"), ha="center", va="center",
                        color="white" if matrix[i, j] >= 0.6 else "black", fontsize=8)
        ax.set(title=f"p={repost:g}", xlabel="q", ylabel="α", xticks=range(len(rewirings)),
               xticklabels=[f"{q:g}" for q in rewirings], yticks=range(len(influences)),
               yticklabels=[f"{alpha:g}" for alpha in influences])
    fig.colorbar(image, ax=flat.tolist(), label="Reference PbS fraction", fraction=0.025, pad=0.025)
    plt_save_and_close(fig, str(output / "reference_pbs_grid"))


def plot_intervention_case(record, output, *, title):
    """Compare two checkpoints of one reference without pooling independent runs."""
    fig, axes = plt_figure(n_row=2, n_col=2, hw_ratio=0.65)
    fig.suptitle(title)
    colors = {"hold": "0.4", "switch": "tab:blue", "bridge": "tab:green", "hide_reposts": "tab:orange"}
    for column, phase in enumerate(("middle", "late")):
        block = record.stages[phase]
        for row, (metric, threshold, ylabel) in enumerate((
            ("separation", cfg.SEPARATION_THRESHOLD, "Opinion separation"),
            ("excess_concordance", cfg.EXCESS_THRESHOLD, "Excess concordance"),
        )):
            ax = axes[row][column]
            for arm, data in block.arms.items():
                ax.plot([point.step for point in data.endpoints],
                        [getattr(point.pathway, metric) for point in data.endpoints],
                        label=label(arm), color=colors[arm], ls="--" if arm == "hold" else "-")
            ax.axhline(threshold, color="0.6", ls=":")
            ax.set(xlabel="Simulation step", ylabel=ylabel)
        axes[0][column].set_title(f"{label(phase)} checkpoint: t={block.checkpoint.step}")
        axes[0][column].legend(fontsize=8)
    plt_save_and_close(fig, str(output / "pbs_intervention_case"))
