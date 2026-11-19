"""Write paired recommendation tables and paper figures from completed records."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.io.json import atomic_json
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.metrics.pathway import PATHWAY_LABELS

from . import design as cfg
from .analysis import (
    OUTCOMES, PEAK_GROUPS, contrast_table, coverage_by_setting, read_study,
    stage_gate, stage_interactions, study_tables,
)

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
}


def label(name):
    return ARM_LABELS.get(name, STAGE_LABELS.get(name, name))


def measurement_label(rows):
    parameters = rows[["epsilon", "history"]].drop_duplicates()
    if len(parameters) != 1:
        raise ValueError("a figure must use one epsilon and history length")
    epsilon, history = parameters.iloc[0]
    return f"ε = {epsilon:g}, history = {int(history)}"


def plot_effects(contrasts, output, order, *, stage_axis=False, stem="paired_effects"):
    pooled = contrasts[(contrasts["setting"] == "all") & (contrasts["repost"] == "all")]
    if pooled.empty:
        return
    fig, axes = plt_figure(n_col=3, hw_ratio=0.78)
    fig.suptitle(measurement_label(pooled))
    key = "stage" if stage_axis else "arm"
    for ax, (metric, title) in zip(axes, METRICS.items()):
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
        ax.set_ylim(len(order) - 0.5, -0.5)
    plt_save_and_close(fig, str(output / stem))


def plot_staged_trajectories(trajectories, output):
    fig, axes = plt_figure(n_col=2, hw_ratio=0.7)
    fig.suptitle(measurement_label(trajectories))
    for ax, metric, title in zip(
        axes, ("separation", "excess_concordance"), ("Opinion separation", "Excess concordance")
    ):
        ax.axhline(0, color="0.6", lw=1)
        for stage in cfg.STAGES:
            sample = trajectories.loc[(trajectories["stage"] == stage) & trajectories["arm"].isin(["hold", "switch"])]
            if sample.empty:
                continue
            paired = sample.pivot(index=["task_id", "relative_step"], columns="arm", values=metric)
            by_step = (paired["switch"] - paired["hold"]).groupby("relative_step")
            mean, sem = by_step.mean(), by_step.sem()
            ax.plot(mean.index, mean, label=f"{label(stage)} (n={sample['task_id'].nunique()})")
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.15)
        ax.set(xlabel="Steps since checkpoint", ylabel=f"Switch − hold\n{title}")
    if axes[0].lines[1:]:
        axes[0].legend()
    plt_save_and_close(fig, str(output / "stage_trajectories"))


def plot_late_exposure(runs, output):
    late = runs[runs["stage"] == "late"]
    if late.empty or late["policy_cross_camp"].isna().all():
        return
    groups = list(late.groupby("repost"))
    fig, axes = plt_figure(n_col=len(groups), hw_ratio=0.65)
    fig.suptitle(measurement_label(late))
    gates = {
        "policy_cross_camp": "Cross-camp",
        "policy_cross_confident": "Within confidence",
        "policy_cross_rewire": "Eligible for rewiring",
    }
    for ax, (repost, rows) in zip(np.asarray(axes, dtype=object).reshape(-1), groups):
        for name in ("hold", "switch", "bridge", "hide_reposts"):
            subset = rows[rows["arm"] == name]
            if subset["policy_cross_camp"].notna().any():
                rates = [float((subset[key] / subset["updates"]).mean()) for key in gates]
                (line,) = ax.plot(list(gates.values()), rates, "o-", label=label(name))
                if name == "bridge":
                    bridge = [
                        float((subset[key.replace("policy_", "bridge_")] / subset["updates"]).mean()) for key in gates
                    ]
                    ax.plot(list(gates.values()), bridge, "o--", color=line.get_color(), label="Bridge inserts")
        ax.tick_params(axis="x", rotation=15)
        ax.set(title=f"Repost probability {repost:g}", ylabel="Posts per update")
        ax.legend()
    plt_save_and_close(fig, str(output / "late_exposure"))


def plot_categories(runs, output, arms, *, stem):
    fig, axes = plt_figure(n_col=3, hw_ratio=0.85)
    fig.suptitle(measurement_label(runs))
    for ax, column, categories, title in zip(
        axes,
        ("outcome", "opinion_shape", "pathway"),
        (OUTCOMES, PEAK_GROUPS, PATHWAY_LABELS),
        ("Confidence components", "Final opinion modes", "First observed passage"),
    ):
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
            yticklabels=list(map(label, arms)),
            title=title,
        )
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt_save_and_close(fig, str(output / stem))


def analyze(directory: Path, output: Path):
    plan, records = read_study(directory)
    if not records:
        raise ValueError(f"no completed recommendation records at {directory}")
    study = plan["profile"]
    runs, trajectories = study_tables(plan, records)
    if runs.empty:
        raise ValueError("completed references contain no eligible stage checkpoints")
    contrasts = contrast_table(runs, study)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    trajectories.to_csv(output / "trajectories.csv", index=False)
    contrasts.to_csv(output / "contrasts.csv", index=False)
    report = {
        "study": study,
        "source_plan": plan["plan_fingerprint"],
        "planned": len(plan["tasks"]),
        "completed": len(records),
        "missing": [task["id"] for task in plan["tasks"] if task["id"] not in records],
        "settings": coverage_by_setting(plan, records),
    }
    setup_paper_params()
    if study == "staged":
        report["stage_gate"] = stage_gate(plan["tasks"], records)
        stage_interactions(runs).to_csv(output / "stage_interactions.csv", index=False)
    elif study == "staged_timing":
        report["latest_fallback_count"] = int(((runs["arm"] == "state") & runs["switch_fallback"]).sum())
        fixed_comparison = contrasts[contrasts["control"] == "fixed"]
        primary = fixed_comparison[
            (fixed_comparison["setting"] == "all") & (fixed_comparison["metric"] == "separation_change")
        ].iloc[0]
        report["primary_outcome"] = {
            "metric": "end_separation",
            "contrast": "state minus fixed",
            "n": int(primary["n"]),
            **{key: float(primary[key]) if pd.notna(primary[key]) else None for key in ("mean", "lo", "hi")},
        }

    for (sensitivity, epsilon, history), sample in runs.groupby(["sensitivity", "epsilon", "history"]):
        destination = output / f"{sensitivity}_e{round(100 * epsilon):02}_kh{int(history)}"
        destination.mkdir(parents=True, exist_ok=True)
        selected = contrasts[
            (contrasts["sensitivity"] == sensitivity) & (contrasts["epsilon"] == epsilon) & (contrasts["history"] == history)
        ]
        if study == "staged":
            plot_effects(selected[selected["arm"] == "switch"], destination, cfg.STAGES, stage_axis=True)
            selected_trajectories = trajectories[
                (trajectories["sensitivity"] == sensitivity) & (trajectories["epsilon"] == epsilon) & (trajectories["history"] == history)
            ]
            plot_staged_trajectories(selected_trajectories, destination)
            plot_late_exposure(sample, destination)
        elif study == "staged_timing":
            plot_effects(selected[selected["control"] == "fixed"], destination, ("state",), stem="state_minus_fixed")
            plot_categories(sample, destination, ("hold", "always_structure", "fixed", "state"), stem="outcomes")
        else:
            arms = tuple(policy for variants in cfg.VARIANTS.values() for policy in variants)
            plot_categories(sample, destination, arms, stem="outcomes")
            within_family = selected[
                (selected["control"].str.endswith("M9"))
                & (selected["arm"].str.startswith("Opinion") == selected["control"].str.startswith("Opinion"))
            ]
            across_families = selected[
                selected["arm"].str.startswith("Opinion") & selected["control"].str.startswith("Structure")
            ]
            plot_effects(within_family, destination, [arm for arm in arms if not arm.endswith("M9")], stem="differences_from_m9")
            plot_effects(across_families, destination, cfg.VARIANTS["Opinion"], stem="opinion_minus_structure")
    atomic_json(output / "coverage.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing plan.json and records/")
    parser.add_argument(
        "--output", type=Path, help="Analysis destination, defaulting to the dataset's analysis directory"
    )
    args = parser.parse_args(argv)
    analyze(args.dataset, args.output or args.dataset / "analysis")


if __name__ == "__main__":
    main()
