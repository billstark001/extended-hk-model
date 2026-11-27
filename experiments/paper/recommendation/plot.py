"""Write checkpoint/timing intervention tables and figures."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from common.io.json import atomic_json
from ehk.domain import TimingSummary, parse_binding, rows_frame, to_wire
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.recommendation_shared import plot as shared_plot

from . import design as cfg
from .analysis import (
    completed_cell_task_ids,
    completed_task_ids,
    contrast_table,
    coverage_by_setting,
    read_study,
    reference_tables,
    stage_gate,
    stage_interactions,
    study_tables,
)


def plot_staged_trajectories(
    trajectories, output, *, title=None, control="hold", arms=("switch",), stem="stage_trajectories"
):
    fig, axes = plt_figure(n_col=2, hw_ratio=0.7)
    fig.suptitle(title or shared_plot.measurement_label(trajectories))
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
                name = shared_plot.label(stage) if arms == ("switch",) else shared_plot.label(arm)
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
    fig.suptitle(title or shared_plot.measurement_label(late))
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
                (line,) = ax.plot(list(gates.values()), rates, "o-", label=shared_plot.label(name))
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
                       label=f"{shared_plot.label(stage)} checkpoints (n={len(sample)})", zorder=3)
        ax.set(xlabel="Reference step", ylabel=ylabel)
    axes[0].legend(fontsize=8)
    plt_save_and_close(fig, str(output / "reference_trajectories"))


def plot_late_opinions(tasks, records, output, *, title):
    tasks = sorted(
        [task for task in tasks if task.id in records and "late" in records[task.id].stages],
        key=lambda task: task.replicate,
    )
    if not tasks:
        return
    arm_names = ("hold", "switch", "bridge", "hide_reposts")
    arms = [arm for arm in arm_names if any(arm in records[task.id].stages["late"].arms for task in tasks)]
    bins = np.linspace(-1, 1, 101)
    matrices = {}
    for arm in arms:
        matrix = np.full((len(tasks), 100), np.nan)
        for i, task in enumerate(tasks):
            variants = records[task.id].stages["late"].arms
            if arm not in variants:
                continue
            opinions = np.asarray(variants[arm].final_opinions, dtype=float)
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
        ax.set(xlabel="Opinion", title=shared_plot.label(arm).replace(" + ", "\n+ "),
               yticks=range(len(tasks)), yticklabels=[task.replicate for task in tasks])
    np.asarray(axes, dtype=object).reshape(-1)[0].set_ylabel("Replicate")
    fig.colorbar(image, ax=np.asarray(axes, dtype=object).reshape(-1).tolist(),
                 label="Agent fraction per opinion bin", fraction=0.025, pad=0.025)
    plt_save_and_close(fig, str(output / "late_opinions"))


def _plot_scope(
    study, sample, trajectories, contrasts, references, selections, destination, *, title, setting="all", repost="all",
):
    destination.mkdir(parents=True, exist_ok=True)
    if study == "staged":
        shared_plot.plot_effects(contrasts[contrasts["arm"] == "switch"], destination, cfg.STAGES,
                     stage_axis=True, setting=setting, repost=repost, title=f"{title}\nSwitch − hold")
        plot_staged_trajectories(trajectories, destination, title=title)
        diagnostics = contrasts[(contrasts["stage"] == "late") & (contrasts["control"] == "switch")]
        shared_plot.plot_effects(diagnostics, destination, ("bridge", "hide_reposts"), stem="mechanism_effects",
                     setting=setting, repost=repost, title=f"{title}\nLate intervention − switch")
        late_trajectories = trajectories[trajectories["stage"] == "late"]
        if not late_trajectories.empty:
            plot_staged_trajectories(late_trajectories, destination, title=title, control="switch",
                                     arms=("bridge", "hide_reposts"), stem="mechanism_trajectories")
        plot_late_exposure(sample, destination, title=title)
        plot_late_exposure(sample, destination, title=title, all_slots=True)
        shared_plot.plot_categories(sample[sample["stage"] == "late"], destination,
                        ("hold", "switch", "bridge", "hide_reposts"), stem="late_outcomes", title=title)
        plot_references(references, selections, destination, title=title)
    elif study == "staged_timing":
        shared_plot.plot_effects(contrasts[contrasts["control"] == "fixed"], destination, ("state",),
                     stem="state_minus_fixed", setting=setting, repost=repost, title=title)
        shared_plot.plot_categories(sample, destination, ("hold", "always_structure", "fixed", "state"),
                        stem="outcomes", title=title)


from experiments.paper.recommendation_shared.storage import write_table


def analyze(directory: Path, output: Path, *, make_figures=True):
    created_at = datetime.now(timezone.utc).isoformat()
    plan, records = read_study(directory)
    if not records:
        raise ValueError(f"no completed recommendation records at {directory}")
    study = plan.profile
    binding_path = directory / "runtime" / "binding.json"
    binding = parse_binding(json.loads(binding_path.read_text(encoding="utf-8"))) if binding_path.is_file() else None
    runs, trajectories = study_tables(plan, records, binding=binding)
    if runs.empty:
        raise ValueError("completed references contain no eligible stage checkpoints")
    contrasts = contrast_table(runs, study)
    references, selections = reference_tables(plan, records) if study == "staged" else (pd.DataFrame(), pd.DataFrame())
    coverage_rows = coverage_by_setting(plan, records)
    coverage = rows_frame(coverage_rows).drop(columns="stages")
    if study == "staged":
        for stage in cfg.STAGES:
            coverage[stage] = [(row.stages or {}).get(stage, 0) for row in coverage_rows]
    complete_ids = completed_cell_task_ids(plan, records)
    completed = completed_task_ids(plan, records)
    complete_runs = runs[runs["task_id"].isin(complete_ids)]
    complete_contrasts = contrast_table(complete_runs, study) if not complete_runs.empty else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("runs", runs), ("trajectories", trajectories), ("contrasts", contrasts),
        ("coverage", coverage), ("complete_cell_contrasts", complete_contrasts),
    ):
        write_table(output, f"{name}.csv.gz", table)
        table.to_csv(output / f"{name}.csv", index=False)
    report = shared_plot.analysis_report(
        directory, plan, records, binding, coverage_rows, coverage, complete_ids, completed, created_at,
    )
    if make_figures:
        setup_paper_params()
        shared_plot.plot_coverage(coverage, output)
    if study == "staged":
        gate = stage_gate(plan.tasks, records)
        status = "pending" if not gate.complete else ("passed" if gate.passed else "not_passed")
        report.stage_gate = replace(gate, status=status)
        interactions = stage_interactions(runs)
        write_table(output, "stage_interactions.csv.gz", interactions)
        write_table(output, "references.csv.gz", references)
        write_table(output, "stage_selection.csv.gz", selections)
        interactions.to_csv(output / "stage_interactions.csv", index=False)
        references.to_csv(output / "references.csv", index=False)
        selections.to_csv(output / "stage_selection.csv", index=False)
    elif study == "staged_timing":
        report.latest_fallback_count = int(((runs["arm"] == "state") & runs["switch_fallback"]).sum())
        fixed_comparison = contrasts[contrasts["control"] == "fixed"]
        primary = fixed_comparison[
            (fixed_comparison["setting"] == "all") & (fixed_comparison["metric"] == "separation_change")
        ].iloc[0]
        report.primary_outcome = TimingSummary(
            metric="end_separation", contrast="state minus fixed", n=int(primary["n"]),
            mean=float(primary["mean"]) if pd.notna(primary["mean"]) else None,
            lo=float(primary["lo"]) if pd.notna(primary["lo"]) else None,
            hi=float(primary["hi"]) if pd.notna(primary["hi"]) else None,
        )

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
                plot_late_opinions([task for task in plan.tasks if task.id in ids], records,
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
        title = f"{shared_plot.measurement_label(pooled)} | {cell_count} complete cells, {len(pooled_ids)} tasks"
        _plot_scope(
            study, pooled, sample_trajectories[sample_trajectories["task_id"].isin(pooled_ids)], selected,
            references[references["task_id"].isin(pooled_ids)] if not references.empty else references,
            selections[selections["task_id"].isin(pooled_ids)] if not selections.empty else selections,
            destination / "complete_cells", title=title,
        )
    atomic_json(output / "coverage.json", to_wire(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing plan.json and analysis.sqlite")
    parser.add_argument(
        "--output", type=Path, help="Analysis destination, defaulting to the dataset's analysis directory"
    )
    parser.add_argument("--tables-only", action="store_true", help="Export tables and provenance without rendering every parameter cell.")
    args = parser.parse_args(argv)
    analyze(args.dataset, args.output or args.dataset / "analysis", make_figures=not args.tables_only)

if __name__ == "__main__":
    main()
