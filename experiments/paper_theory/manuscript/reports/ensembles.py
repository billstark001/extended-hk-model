"""Manuscript ensemble controls and long-horizon clocks for the full paper design."""

from __future__ import annotations

import json
from pathlib import Path

from common.io.json import atomic_json, sha
from common.io.tables import table
from experiments.paper_theory.manuscript.reports.tables import number, read_table

from ...micro_validation.analyze.controls import ensemble_grid, mechanism_contrasts, population
from ...release_clocks.analyze.review import clock_estimates, crossing_ranges
from ..figures.ensembles import plots
from .dataset import gather


def review(output_root, label, baseline_label, out):
    """Write secondary ensemble evidence and figures using the full production control cohorts plus a source-matched baseline."""
    out.mkdir(parents=True, exist_ok=True)
    root, baseline = (output_root / label, output_root / baseline_label)
    plan, micro, clock, audit = gather(root, baseline, out)
    estimates = clock_estimates(root, out)
    ensemble_controls = ensemble_grid(micro, plan["analysis_protocol"], out)
    contrast = mechanism_contrasts(micro, plan["analysis_protocol"], out)
    pop, pop_aggregate, selected = population(micro, root, plan["analysis_protocol"], out)
    crossings = crossing_ranges(root, out)
    primary_micro = read_table(root / "analysis" / "micro" / "trajectory_errors.csv")
    table(out / "basin_grid.csv", [r for r in primary_micro if r["role"] == "basin_grid"])
    table(out / "holdout_grid.csv", [r for r in primary_micro if r["role"] == "holdout_grid"])
    ensemble = read_table(root / "analysis" / "micro" / "ensemble_distribution_errors.csv")
    basin_ensemble = [
        r
        for r in ensemble
        if number(r, "n") == 500
        and number(r, "b") == 161
        and (number(r, "quadrature") == 33)
        and (number(r, "ratio") == 10)
    ]
    table(out / "basin_conditional_ensembles.csv", basin_ensemble)
    clock_rows = [r["row"] for r in clock]
    extensions = [r for r in clock_rows if r["role"] == "extension"]
    table(out / "extended_platforms.csv", extensions)
    table(out / "unconfirmed_clocks.csv", [r for r in clock_rows if not r["energy_confirmed"]])
    table(
        out / "state_energy_separation.csv",
        [
            {
                "id": r["id"],
                **r["meta"],
                "energy_confirmed": r["row"]["energy_confirmed"],
                "joint_confirmed": r["row"]["joint_confirmed"],
                "actual_tau": r["row"]["actual_tau"],
                **r["diagnostics"],
            }
            for r in clock
        ],
    )
    summary = {
        "audit": audit,
        "review_source_sha256": sha(Path(__file__)),
        "method": "incremental secondary review; original raw data and primary analysis preserved",
        "input_tables_sha256": {str(p.relative_to(root)): sha(p) for p in sorted((root / "analysis").rglob("*.csv"))},
        "clock_prediction": {
            "cases": len(estimates),
            "pair_within_mean_interval": sum((r["pair_within_mean_interval"] for r in estimates)),
            **{
                c: {
                    "min_relative_error": min((r["relative_error"] for r in estimates if r["channel"] == c)),
                    "max_relative_error": max((r["relative_error"] for r in estimates if r["channel"] == c)),
                    "covered": sum((r["pair_within_mean_interval"] for r in estimates if r["channel"] == c)),
                }
                for c in ("node", "edge")
            },
        },
        "recommendation_contrasts": contrast,
        "population_equal_repeats": pop,
        "conditional_ensemble_grid": ensemble_controls,
        "population_aggregate": pop_aggregate,
        "crossing_ranges": crossings,
        "long_clocks": {
            "total": len(clock),
            "energy_confirmed": sum((r["energy_confirmed"] for r in clock_rows)),
            "joint_confirmed": sum((r["joint_confirmed"] for r in clock_rows)),
            "extensions_energy_confirmed": sum((r["energy_confirmed"] for r in extensions)),
            "extensions_joint_confirmed": sum((r["joint_confirmed"] for r in extensions)),
            "max_energy_budget_residual": max((abs(r["budget_residual"]) for r in clock_rows)),
            "max_tail_relative_change_confirmed": max(
                (
                    r[c + "_tail_relative_change"] or 0
                    for r in clock_rows
                    if r["energy_confirmed"]
                    for c in ("rho", "edge")
                )
            ),
            "unconfirmed_ids": [r["id"] for r in clock_rows if not r["energy_confirmed"]],
            "progress_order_inconsistent": sum((r["order_consistent"] is False for r in clock_rows)),
            "nonmonotone_release": sum((r[c + "_nonmonotone_release"] for r in clock_rows for c in ("rho", "edge"))),
        },
    }
    plots(micro, clock, root, out, estimates, selected, ensemble_controls)
    atomic_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "audit": audit,
                "clock_prediction": summary["clock_prediction"],
                "recommendation_contrasts": contrast,
                "long_clocks": summary["long_clocks"],
            },
            indent=2,
        )
    )
