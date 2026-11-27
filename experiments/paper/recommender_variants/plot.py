"""Write full-trajectory recommendation-family and factorial tables and figures."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.io.json import atomic_json
from ehk.domain import ArmTotals, parse_binding, rows_frame, to_wire
from common.plotting import setup_paper_params
from experiments.paper.recommendation_shared import plot as shared_plot

from . import design as cfg
from .analysis import (
    FACTORIAL_CONTRASTS,
    completed_cell_task_ids,
    completed_task_ids,
    contrast_table,
    coverage_by_setting,
    opinion_random_factorial,
    read_study,
    study_tables,
    variant_arm_coverage,
)


def _plot_scope(
    sample, contrasts, destination, *, title, setting="all", repost="all",
    factorial=None,
):
    destination.mkdir(parents=True, exist_ok=True)
    arms = tuple(dict.fromkeys([
        *(policy for variants in cfg.VARIANTS.values() for policy in variants), *sample["arm"],
    ]))
    shared_plot.plot_categories(sample, destination, arms, stem="outcomes", title=title)
    same_family = contrasts["arm"].str.startswith("Opinion") == contrasts["control"].str.startswith("Opinion")
    within_family = contrasts[contrasts["control"].str.endswith("M9") & same_family]
    across_families = contrasts[
        contrasts["arm"].str.startswith("Opinion") & contrasts["control"].str.startswith("Structure")
    ]
    shared_plot.plot_effects(within_family, destination, [arm for arm in arms if not arm.endswith("M9")],
                 stem="differences_from_m9", setting=setting, repost=repost, title=title)
    shared_plot.plot_effects(across_families, destination, cfg.VARIANTS["Opinion"],
                 stem="opinion_minus_structure", setting=setting, repost=repost, title=title)
    if factorial is not None and not factorial.empty:
        shared_plot.plot_effects(factorial.rename(columns={"contrast": "arm"}), destination, FACTORIAL_CONTRASTS,
                     stem="opinion_random_factorial", setting=setting, repost=repost, title=title)


from experiments.paper.recommendation_shared.storage import write_table


def analyze(directory: Path, output: Path, *, make_figures=True):
    created_at = datetime.now(timezone.utc).isoformat()
    plan, records = read_study(directory)
    if not records:
        raise ValueError(f"no completed recommendation records at {directory}")
    study = plan.profile
    if study != cfg.STUDY:
        raise ValueError("recommender-variant analysis requires its own study profile")
    binding_path = directory / "runtime" / "binding.json"
    binding = parse_binding(json.loads(binding_path.read_text(encoding="utf-8"))) if binding_path.is_file() else None
    runs, trajectories = study_tables(plan, records, binding=binding)
    if runs.empty:
        raise ValueError("completed records contain no recommendation-policy arms")
    contrasts = contrast_table(runs)
    factorial = opinion_random_factorial(plan, runs)
    arm_coverage = variant_arm_coverage(plan, records, binding=binding)
    coverage_rows = coverage_by_setting(plan, records)
    coverage = rows_frame(coverage_rows).drop(columns="stages")
    complete_ids = completed_cell_task_ids(plan, records)
    completed = completed_task_ids(plan, records)
    complete_runs = runs[runs["task_id"].isin(complete_ids)]
    complete_contrasts = contrast_table(complete_runs) if not complete_runs.empty else pd.DataFrame()
    complete_factorial = opinion_random_factorial(plan, complete_runs) if not complete_runs.empty else pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("runs", runs), ("trajectories", trajectories), ("contrasts", contrasts),
        ("coverage", coverage), ("complete_cell_contrasts", complete_contrasts),
    ):
        write_table(output, f"{name}.csv.gz", table)
        table.to_csv(output / f"{name}.csv", index=False)
    for name, table in (
        ("arm_coverage", arm_coverage), ("opinion_random_factorial", factorial),
        ("complete_cell_opinion_random_factorial", complete_factorial),
    ):
        write_table(output, f"{name}.csv.gz", table)
        table.to_csv(output / f"{name}.csv", index=False)
    report = shared_plot.analysis_report(
        directory, plan, records, binding, coverage_rows, coverage, complete_ids, completed, created_at,
    )
    report.arms = tuple(ArmTotals(arm=row.arm, planned=int(row.planned), completed=int(row.completed))
                        for row in arm_coverage.groupby("arm")[["planned", "completed"]].sum().reset_index().itertuples(index=False))
    if make_figures:
        setup_paper_params()
        shared_plot.plot_coverage(coverage, output)
    groups = runs.groupby(["sensitivity", "epsilon", "history"]) if make_figures else ()
    for (sensitivity, epsilon, history), sample in groups:
        destination = output / f"{sensitivity}_e{round(100 * epsilon):02}_kh{int(history)}"
        common = ((contrasts["sensitivity"] == sensitivity) & (contrasts["epsilon"] == epsilon)
                  & (contrasts["history"] == history))
        for (setting, repost), cell in sample.groupby(["setting", "repost"]):
            ids = set(cell["task_id"])
            row = coverage[(coverage["setting"] == setting) & (coverage["repost"] == repost)].iloc[0]
            title = (f"α={row['influence']:g}, q={row['rewiring']:g}, p={repost:g}, "
                     f"ε={epsilon:g}, history={int(history)} | {len(ids)}/{int(row['planned'])} tasks")
            title += f", {int(row['completed'])} complete"
            selected = contrasts[common & (contrasts["setting"] == setting) & (contrasts["repost"] == repost)]
            _plot_scope(
                cell, selected,
                destination / f"{setting}_p{round(100 * repost):02}",
                title=title, setting=setting, repost=repost,
                factorial=factorial[
                    (factorial["sensitivity"] == sensitivity) & (factorial["epsilon"] == epsilon)
                    & (factorial["history"] == history) & (factorial["setting"] == setting)
                    & (factorial["repost"] == repost)
                ],
            )
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
            pooled, selected,
            destination / "complete_cells", title=title,
            factorial=complete_factorial[
                (complete_factorial["sensitivity"] == sensitivity) & (complete_factorial["epsilon"] == epsilon)
                & (complete_factorial["history"] == history)
            ] if not complete_factorial.empty else complete_factorial,
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
