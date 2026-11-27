"""Shared presentation helpers for recorded recommendation-study metrics."""

import matplotlib as mpl
import numpy as np
import pandas as pd

from .storage import task_digest, dataset_raw_root
from ehk.domain import AnalysisReport, PooledCoverage
from common.plotting import plt_figure, plt_save_and_close
from common.plotting.figures import save
from ehk.metrics.pathway import PATHWAY_LABELS

from .analysis import OUTCOMES, PEAK_GROUPS

METRICS = {
    "separation_change": "Change in opinion separation",
    "excess_change": "Change in excess concordance",
    "cross_camp_per_update": "Cross-camp posts per update",
}

STAGE_LABELS = {"early": "Early", "middle": "Middle", "late": "Late"}

ARM_LABELS = {
    "hold": "Keep OpinionM9",
    "switch": "Switch to StructureM9",
    "bridge": "StructureM9 + bridge",
    "hide_reposts": "StructureM9 + hide reposts",
    "always_structure": "Always StructureM9",
    "state": "State trigger",
    "fixed": "Fixed step",
    "zeta_at_score_noise_0": "ζ: 4 − 1, score σ = 0",
    "zeta_at_score_noise_2": "ζ: 4 − 1, score σ = 2",
    "score_noise_at_zeta_1": "Score σ: 2 − 0, ζ = 1",
    "score_noise_at_zeta_4": "Score σ: 2 − 0, ζ = 4",
    "interaction": "ζ effect at σ = 2 − at σ = 0",
}


def label(name):
    return ARM_LABELS.get(name, STAGE_LABELS.get(name, name))


def measurement_label(rows):
    parameters = rows[["epsilon", "history"]].drop_duplicates()
    if len(parameters) != 1:
        raise ValueError("a figure must use one epsilon and history length")
    epsilon, history = parameters.iloc[0]
    return f"ε = {epsilon:g}, history = {int(history)}"


def plot_effects(
    contrasts, output, order, *, stage_axis=False, stem="paired_effects", setting="all", repost="all", title=None
):
    pooled = contrasts[(contrasts["setting"] == setting) & (contrasts["repost"] == repost)]
    if pooled.empty or not (pooled["n"] > 0).any():
        return
    fig, axes = plt_figure(n_col=3, hw_ratio=0.78, sharey=True)
    fig.suptitle(title or measurement_label(pooled))
    key = "stage" if stage_axis else "arm"
    for index, (ax, (metric, title)) in enumerate(zip(axes, METRICS.items())):
        subset = pooled[(pooled["metric"] == metric) & (pooled["n"] > 0)].set_index(key)
        ax.axvline(0, color="0.6", lw=1)
        for i, item in enumerate(order):
            if item not in subset.index:
                continue
            row = subset.loc[item]
            ax.plot(row["mean"], i, "o", color="tab:blue")
            if pd.notna(row["lo"]):
                ax.hlines(i, row["lo"], row["hi"], color="tab:blue")
            ax.text(0.98, i + 0.15, f"n={int(row['n'])}", transform=ax.get_yaxis_transform(), ha="right", fontsize=8)
        ax.set(yticks=range(len(order)), yticklabels=list(map(label, order)), xlabel=title)
        ax.tick_params(labelleft=index == 0)
        ax.set_ylim(len(order) - 0.5, -0.5)
    plt_save_and_close(fig, str(output / stem))


def plot_categories(runs, output, arms, *, stem, title=None):
    if runs.empty:
        return
    arms = [arm for arm in arms if arm in set(runs["arm"])]
    arm_labels = {arm: label(arm) for arm in arms}
    if "recommender" in runs:
        for arm, block in runs[runs["recommender"] == "OpinionRandom"].groupby("arm"):
            factors = block[["zeta", "score_noise"]].drop_duplicates()
            if len(factors) != 1:
                raise ValueError(f"ambiguous OpinionRandom configuration: {arm}")
            zeta, noise = factors.iloc[0]
            arm_labels[arm] = f"OpinionRandom (ζ={zeta:g}, score σ={noise:g})"
    fig, axes = plt_figure(n_col=3, hw_ratio=0.85, sharey=True)
    fig.suptitle(title or measurement_label(runs))
    for index, (ax, column, categories, title) in enumerate(zip(
        axes,
        ("outcome", "opinion_shape", "pathway"),
        (OUTCOMES, PEAK_GROUPS, PATHWAY_LABELS),
        ("Confidence components", "Final opinion modes", "Full-history first passage"),
    )):
        data = pd.crosstab(runs["arm"], runs[column], normalize="index").reindex(index=arms, columns=categories)
        data = data.fillna(0)
        left = np.zeros(len(arms))
        for category in categories:
            counts = data[category].to_numpy()
            ax.barh(np.arange(len(arms)), counts, left=left, label=category.replace("_", " "))
            left += counts
        ax.set(
            xlim=(0, 1),
            xlabel="Fraction of runs",
            yticks=np.arange(len(arms)),
            yticklabels=[f"{arm_labels[arm]} (n={runs.loc[runs['arm'] == arm, 'task_id'].nunique()})" for arm in arms],
            title=title,
        )
        ax.tick_params(labelleft=index == 0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axes[0].invert_yaxis()
    plt_save_and_close(fig, str(output / stem))


def plot_coverage(coverage, output):
    settings = coverage.drop_duplicates("setting").set_index("setting")
    reposts = sorted(coverage["repost"].unique())
    fractions = np.full((len(settings), len(reposts)), np.nan)
    counts = {}
    for row in coverage.itertuples():
        coordinate = settings.index.get_loc(row.setting), reposts.index(row.repost)
        fractions[coordinate] = row.completed / row.planned
        counts[coordinate] = f"{row.completed}/{row.planned}"
    fig, ax = plt_figure(hw_ratio=max(0.4, 0.06 * len(settings) + 0.15), total_width=10)
    cmap = mpl.colormaps["Greens"].with_extremes(bad="0.9")
    image = ax.imshow(np.ma.masked_invalid(fractions), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(len(settings)):
        for j in range(len(reposts)):
            color = "white" if fractions[i, j] >= 0.6 else "black"
            ax.text(j, i, counts.get((i, j), "—"), ha="center", va="center", color=color)
    labels = [
        f"α={row.influence:g}, q={row.rewiring:g}, ε={row.epsilon:g}, history={int(row.history)}"
        for row in settings.itertuples()
    ]
    ax.set(xticks=range(len(reposts)), xticklabels=[f"p={p:g}" for p in reposts],
           yticks=range(len(settings)), yticklabels=labels, title="Completed paired tasks / planned tasks")
    fig.colorbar(image, ax=ax, label="Completion fraction", fraction=0.035, pad=0.03)
    plt_save_and_close(fig, str(output / "coverage"))


def _save_panel(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    save(fig, path)


def analysis_report(directory, plan, records, binding, coverage_rows, coverage, complete_ids, completed, created_at):
    """Record coverage and provenance without choosing a study's scientific contrasts."""
    return AnalysisReport(study=plan.profile, created_at_utc=created_at,
        source_directory=str(directory.resolve()), source_plan=plan.plan_fingerprint, binding=binding,
        record_sha256={task.id: task_digest(directory,task,plan) for task in plan.tasks if task.id in records},
        raw_root=str(dataset_raw_root(directory,plan)), planned=len(plan.tasks), available_records=len(records),
        completed=len(completed), complete=len(completed)==len(plan.tasks),
        missing=tuple(task.id for task in plan.tasks if task.id not in completed), settings=tuple(coverage_rows),
        pooled_figures=PooledCoverage(rule="completed_parameter_cells_only",tasks=len(complete_ids),
                                     cells=int((coverage["completed"]==coverage["planned"]).sum())))
