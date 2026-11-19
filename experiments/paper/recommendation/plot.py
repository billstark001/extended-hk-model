"""Write paired recommendation tables and paper figures from completed records."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from common.io.json import atomic_json, sha
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.metrics.pathway import PATHWAY_LABELS

from . import design as cfg
from .analysis import (
    OUTCOMES, PEAK_GROUPS, completed_cell_task_ids, contrast_table, coverage_by_setting, read_study,
    reference_tables, stage_gate, stage_interactions, study_tables,
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


def plot_effects(
    contrasts, output, order, *, stage_axis=False, stem="paired_effects", setting="all", repost="all", title=None
):
    pooled = contrasts[(contrasts["setting"] == setting) & (contrasts["repost"] == repost)]
    if pooled.empty:
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


def plot_staged_trajectories(
    trajectories, output, *, title=None, control="hold", arms=("switch",), stem="stage_trajectories"
):
    fig, axes = plt_figure(n_col=2, hw_ratio=0.7)
    fig.suptitle(title or measurement_label(trajectories))
    for ax, metric, title in zip(
        axes, ("separation", "excess_concordance"), ("Opinion separation", "Excess concordance")
    ):
        ax.axhline(0, color="0.6", lw=1)
        for stage in cfg.STAGES:
            for arm in arms:
                sample = trajectories.loc[
                    (trajectories["stage"] == stage) & trajectories["arm"].isin([control, arm])
                ]
                if sample.empty or not {control, arm} <= set(sample["arm"]):
                    continue
                paired = sample.pivot(index=["task_id", "relative_step"], columns="arm", values=metric)
                differences = (paired[arm] - paired[control]).dropna()
                by_step = differences.groupby("relative_step")
                mean, sem = by_step.mean(), by_step.sem()
                name = label(stage) if arms == ("switch",) else label(arm)
                ax.plot(mean.index, mean, label=f"{name} (n={differences.index.get_level_values('task_id').nunique()})")
                ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.15)
        comparison = "Switch − hold" if control == "hold" else "Intervention − switch"
        ax.set(xlabel="Steps since checkpoint", ylabel=f"{comparison}\n{title}")
    if axes[0].lines[1:]:
        axes[0].legend()
    plt_save_and_close(fig, str(output / stem))


def plot_late_exposure(runs, output, *, title=None, all_slots=False):
    late = runs[runs["stage"] == "late"]
    if late.empty:
        return
    groups = list(late.groupby("repost"))
    fig, axes = plt_figure(n_col=len(groups), hw_ratio=0.65)
    fig.suptitle(title or measurement_label(late))
    prefix = "" if all_slots else "non_exploration_"
    gates = {
        f"{prefix}cross_camp": "Cross-camp",
        f"{prefix}cross_confident": "Within confidence",
        f"{prefix}cross_rewire": "Eligible for rewiring",
    }
    for ax, (repost, rows) in zip(np.asarray(axes, dtype=object).reshape(-1), groups):
        for name in ("hold", "switch", "bridge", "hide_reposts"):
            subset = rows[rows["arm"] == name]
            if not subset.empty:
                rates = [float((subset[key] / subset["updates"]).mean()) for key in gates]
                (line,) = ax.plot(list(gates.values()), rates, "o-", label=label(name))
                if name == "bridge":
                    bridge = [
                        float((subset[f"bridge_{key.removeprefix(prefix)}"] / subset["updates"]).mean())
                        for key in gates
                    ]
                    ax.plot(list(gates.values()), bridge, "o--", color=line.get_color(), label="Bridge inserts")
        ax.tick_params(axis="x", rotation=15)
        scope = "All recommendation slots" if all_slots else "Policy + bridge slots"
        ax.set(title=f"Repost probability {repost:g}", ylabel=f"{scope}\nPosts per update")
        ax.legend()
    plt_save_and_close(fig, str(output / ("all_slot_exposure" if all_slots else "late_exposure")))


def plot_categories(runs, output, arms, *, stem, title=None):
    if runs.empty:
        return
    arms = [arm for arm in arms if arm in set(runs["arm"])]
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
            yticklabels=[f"{label(arm)} (n={runs.loc[runs['arm'] == arm, 'task_id'].nunique()})" for arm in arms],
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


def plot_references(references, selections, output, *, title):
    if references.empty:
        return
    fig, axes = plt_figure(n_col=2, hw_ratio=0.7)
    fig.suptitle(title)
    colors = {"early": "tab:green", "middle": "tab:orange", "late": "tab:red"}
    for ax, metric, threshold, ylabel in zip(
        axes, ("separation", "excess_concordance"),
        (cfg.SEPARATION_THRESHOLD, cfg.EXCESS_THRESHOLD), ("Opinion separation", "Excess concordance"),
    ):
        by_step = references.groupby("step")[metric]
        mean, sem = by_step.mean(), by_step.sem()
        ax.plot(mean.index, mean, color="tab:blue", label="Reference mean")
        ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.15, color="tab:blue")
        ax.axhline(threshold, color="0.5", ls="--", label=f"Threshold {threshold:g}")
        for stage in cfg.STAGES:
            sample = selections.loc[selections["stage"] == stage] if not selections.empty else selections
            if sample.empty:
                continue
            ax.scatter(sample["step"], sample[metric], color=colors[stage], s=18, alpha=0.65,
                       label=f"{label(stage)} checkpoints (n={len(sample)})", zorder=3)
        ax.set(xlabel="Reference step", ylabel=ylabel)
    axes[0].legend(fontsize=8)
    plt_save_and_close(fig, str(output / "reference_trajectories"))


def plot_late_opinions(tasks, records, output, *, title):
    tasks = sorted(
        [task for task in tasks if task["id"] in records and "late" in records[task["id"]]["stages"]],
        key=lambda task: task["replicate"],
    )
    if not tasks:
        return
    arm_names = ("hold", "switch", "bridge", "hide_reposts")
    arms = [arm for arm in arm_names if any(arm in records[task["id"]]["stages"]["late"]["arms"] for task in tasks)]
    bins = np.linspace(-1, 1, 101)
    matrices = {}
    for arm in arms:
        matrix = np.full((len(tasks), 100), np.nan)
        for i, task in enumerate(tasks):
            variants = records[task["id"]]["stages"]["late"]["arms"]
            if arm not in variants:
                continue
            opinions = np.asarray(variants[arm]["final_opinions"], dtype=float)
            if not opinions.size or not np.isfinite(opinions).all() or np.any(np.abs(opinions) > 1):
                raise ValueError(f"invalid final opinions for {task['id']}/{arm}")
            matrix[i] = np.histogram(opinions, bins=bins)[0] / opinions.size
        matrices[arm] = matrix
    fig, axes = plt_figure(n_col=len(arms), hw_ratio=max(0.9, len(tasks) * 0.08))
    fig.suptitle(title)
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    for ax, arm in zip(np.asarray(axes, dtype=object).reshape(-1), arms):
        image = ax.imshow(matrices[arm], origin="upper", aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                          extent=(-1, 1, len(tasks) - 0.5, -0.5), interpolation="nearest")
        ax.set(xlabel="Opinion", title=label(arm).replace(" + ", "\n+ "),
               yticks=range(len(tasks)), yticklabels=[task["replicate"] for task in tasks])
    np.asarray(axes, dtype=object).reshape(-1)[0].set_ylabel("Replicate")
    fig.colorbar(image, ax=np.asarray(axes, dtype=object).reshape(-1).tolist(),
                 label="Agent fraction per opinion bin", fraction=0.025, pad=0.025)
    plt_save_and_close(fig, str(output / "late_opinions"))


def _plot_scope(
    study, sample, trajectories, contrasts, references, selections, destination, *, title, setting="all", repost="all"
):
    destination.mkdir(parents=True, exist_ok=True)
    if study == "staged":
        plot_effects(contrasts[contrasts["arm"] == "switch"], destination, cfg.STAGES,
                     stage_axis=True, setting=setting, repost=repost, title=f"{title}\nSwitch − hold")
        plot_staged_trajectories(trajectories, destination, title=title)
        diagnostics = contrasts[(contrasts["stage"] == "late") & (contrasts["control"] == "switch")]
        plot_effects(diagnostics, destination, ("bridge", "hide_reposts"), stem="mechanism_effects",
                     setting=setting, repost=repost, title=f"{title}\nLate intervention − switch")
        late_trajectories = trajectories[trajectories["stage"] == "late"]
        if not late_trajectories.empty:
            plot_staged_trajectories(late_trajectories, destination, title=title, control="switch",
                                     arms=("bridge", "hide_reposts"), stem="mechanism_trajectories")
        plot_late_exposure(sample, destination, title=title)
        plot_late_exposure(sample, destination, title=title, all_slots=True)
        plot_categories(sample[sample["stage"] == "late"], destination,
                        ("hold", "switch", "bridge", "hide_reposts"), stem="late_outcomes", title=title)
        plot_references(references, selections, destination, title=title)
    elif study == "staged_timing":
        plot_effects(contrasts[contrasts["control"] == "fixed"], destination, ("state",),
                     stem="state_minus_fixed", setting=setting, repost=repost, title=title)
        plot_categories(sample, destination, ("hold", "always_structure", "fixed", "state"),
                        stem="outcomes", title=title)
    else:
        arms = tuple(policy for variants in cfg.VARIANTS.values() for policy in variants)
        plot_categories(sample, destination, arms, stem="outcomes", title=title)
        same_family = contrasts["arm"].str.startswith("Opinion") == contrasts["control"].str.startswith("Opinion")
        within_family = contrasts[contrasts["control"].str.endswith("M9") & same_family]
        across_families = contrasts[
            contrasts["arm"].str.startswith("Opinion") & contrasts["control"].str.startswith("Structure")
        ]
        plot_effects(within_family, destination, [arm for arm in arms if not arm.endswith("M9")],
                     stem="differences_from_m9", setting=setting, repost=repost, title=title)
        plot_effects(across_families, destination, cfg.VARIANTS["Opinion"],
                     stem="opinion_minus_structure", setting=setting, repost=repost, title=title)


def analyze(directory: Path, output: Path, *, make_figures=True):
    created_at = datetime.now(timezone.utc).isoformat()
    plan, records = read_study(directory)
    if not records:
        raise ValueError(f"no completed recommendation records at {directory}")
    study = plan["profile"]
    runs, trajectories = study_tables(plan, records)
    if runs.empty:
        raise ValueError("completed references contain no eligible stage checkpoints")
    contrasts = contrast_table(runs, study)
    references, selections = reference_tables(plan, records) if study == "staged" else (pd.DataFrame(), pd.DataFrame())
    coverage_rows = coverage_by_setting(plan, records)
    coverage = pd.DataFrame([
        {**{key: value for key, value in row.items() if key != "stages"}, **row.get("stages", {})}
        for row in coverage_rows
    ])
    complete_ids = completed_cell_task_ids(plan, records)
    complete_runs = runs[runs["task_id"].isin(complete_ids)]
    complete_contrasts = contrast_table(complete_runs, study) if not complete_runs.empty else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("runs", runs), ("trajectories", trajectories), ("contrasts", contrasts),
        ("coverage", coverage), ("complete_cell_contrasts", complete_contrasts),
    ):
        table.to_csv(output / f"{name}.csv", index=False)
    report = {
        "study": study,
        "created_at_utc": created_at,
        "source_directory": str(directory.resolve()),
        "source_plan": plan["plan_fingerprint"],
        "binding": json.loads((directory / "runtime" / "binding.json").read_text())
        if (directory / "runtime" / "binding.json").is_file() else None,
        "snapshot": json.loads((directory / "snapshot.json").read_text())
        if (directory / "snapshot.json").is_file() else None,
        "record_sha256": {identifier: sha(directory / "records" / (identifier + ".json.gz")) for identifier in records},
        "planned": len(plan["tasks"]),
        "completed": len(records),
        "complete": len(records) == len(plan["tasks"]),
        "missing": [task["id"] for task in plan["tasks"] if task["id"] not in records],
        "settings": coverage_rows,
        "pooled_figures": {"rule": "completed_parameter_cells_only", "tasks": len(complete_ids),
                           "cells": int((coverage["completed"] == coverage["planned"]).sum())},
    }
    if make_figures:
        setup_paper_params()
        plot_coverage(coverage, output)
    if study == "staged":
        gate = stage_gate(plan["tasks"], records)
        status = "pending" if not gate["complete"] else ("passed" if gate["passed"] else "not_passed")
        report["stage_gate"] = {**gate, "status": status}
        stage_interactions(runs).to_csv(output / "stage_interactions.csv", index=False)
        references.to_csv(output / "references.csv", index=False)
        selections.to_csv(output / "stage_selection.csv", index=False)
    elif study == "staged_timing":
        report["latest_fallback_count"] = int(((runs["arm"] == "state") & runs["switch_fallback"]).sum())
        fixed_comparison = contrasts[contrasts["control"] == "fixed"]
        primary = fixed_comparison[
            (fixed_comparison["setting"] == "all") & (fixed_comparison["metric"] == "separation_change")
        ].iloc[0]
        report["primary_outcome"] = {
            "metric": "end_separation", "contrast": "state minus fixed", "n": int(primary["n"]),
            **{key: float(primary[key]) if pd.notna(primary[key]) else None for key in ("mean", "lo", "hi")},
        }

    groups = runs.groupby(["sensitivity", "epsilon", "history"]) if make_figures else ()
    for (sensitivity, epsilon, history), sample in groups:
        destination = output / f"{sensitivity}_e{round(100 * epsilon):02}_kh{int(history)}"
        common = ((contrasts["sensitivity"] == sensitivity) & (contrasts["epsilon"] == epsilon)
                  & (contrasts["history"] == history))
        sample_ids = set(sample["task_id"])
        sample_trajectories = trajectories[trajectories["task_id"].isin(sample_ids)]
        for (setting, repost), cell in sample.groupby(["setting", "repost"]):
            ids = set(cell["task_id"])
            row = coverage[(coverage["setting"] == setting) & (coverage["repost"] == repost)].iloc[0]
            title = (f"α={row['influence']:g}, q={row['rewiring']:g}, p={repost:g}, "
                     f"ε={epsilon:g}, history={int(history)} | {len(ids)}/{int(row['planned'])} tasks")
            selected = contrasts[common & (contrasts["setting"] == setting) & (contrasts["repost"] == repost)]
            _plot_scope(
                study, cell, sample_trajectories[sample_trajectories["task_id"].isin(ids)], selected,
                references[references["task_id"].isin(ids)] if not references.empty else references,
                selections[selections["task_id"].isin(ids)] if not selections.empty else selections,
                destination / f"{setting}_p{round(100 * repost):02}",
                title=title, setting=setting, repost=repost,
            )
            if study == "staged":
                plot_late_opinions([task for task in plan["tasks"] if task["id"] in ids], records,
                                   destination / f"{setting}_p{round(100 * repost):02}", title=title)
        pooled = sample[sample["task_id"].isin(complete_ids)]
        if pooled.empty:
            continue
        pooled_ids = set(pooled["task_id"])
        selected = complete_contrasts[
            (complete_contrasts["sensitivity"] == sensitivity) & (complete_contrasts["epsilon"] == epsilon)
            & (complete_contrasts["history"] == history)
        ]
        cell_count = len(pooled[["setting", "repost"]].drop_duplicates())
        title = f"{measurement_label(pooled)} | {cell_count} complete cells, {len(pooled_ids)} tasks"
        _plot_scope(
            study, pooled, sample_trajectories[sample_trajectories["task_id"].isin(pooled_ids)], selected,
            references[references["task_id"].isin(pooled_ids)] if not references.empty else references,
            selections[selections["task_id"].isin(pooled_ids)] if not selections.empty else selections,
            destination / "complete_cells", title=title,
        )
    atomic_json(output / "coverage.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing plan.json and records/")
    parser.add_argument(
        "--output", type=Path, help="Analysis destination, defaulting to the dataset's analysis directory"
    )
    parser.add_argument("--tables-only", action="store_true", help="Export tables and provenance without rendering every parameter cell.")
    args = parser.parse_args(argv)
    analyze(args.dataset, args.output or args.dataset / "analysis", make_figures=not args.tables_only)


if __name__ == "__main__":
    main()
