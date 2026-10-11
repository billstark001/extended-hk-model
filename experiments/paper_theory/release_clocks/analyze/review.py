"""Clock plateau sensitivity, ensemble coverage and sampled crossing manuscript evidence."""

from __future__ import annotations

import collections
import math

from common.io.tables import table
from experiments.paper_theory.manuscript.reports.tables import read_table
from experiments.paper_theory.reporting import clock_group, energy_only_tail, number, raw, read_csv, write_csv


def review_clocks(root, out, summary, clocks):
    """Add reviewed clock, energy-only sensitivity and crossing evidence to summary; distinguish primary joint plateau checks from post hoc energy checks."""
    for row in clocks:
        verified = number(row, "delta50") is not None
        row["primary_joint_plateau"] = verified
        row["energy_only_added"] = False
        row["energy_only_final_pass"] = verified
        if not verified:
            record = raw(root, "clocks", row["id"])
            passed, stable = energy_only_tail(record)
            row["energy_only_final_pass"] = passed
            row["energy_only_stable_steps"] = stable
            row["energy_only_added"] = passed
            row.update(
                {
                    k: record["result"]["diagnostics"][k]
                    for k in ("final_state_l1_rate", "final_node_energy_rate", "final_edge_energy_rate")
                }
            )
        a, b = (number(row, "rho_T50"), number(row, "edge_T50"))
        row["finite_delta50"] = math.log(b / a) if a and b else None
        row["q0_roundoff_tie"] = (
            float(row["ratio"]) == 0 and row["finite_delta50"] is not None and (abs(row["finite_delta50"]) < 1e-09)
        )
        for channel in ("rho", "edge"):
            original, alternative = (number(row, channel + "_T50"), number(row, channel + "_T50_tail20"))
            row[channel + "_tail_relative_change"] = (
                abs(alternative / original - 1) if original and alternative else None
            )
    write_csv(out / "clocks_reviewed.csv", clocks)
    reviewed_crossings = []
    for row in read_csv(root, "clocks", "crossing_intervals.csv"):
        between = [
            r
            for r in clocks
            if clock_group(r) == clock_group(row)
            and float(row["left_ratio"]) < float(r["ratio"]) < float(row["right_ratio"])
        ]
        row["skipped_design_points"] = len(between)
        row["adjacent_in_design"] = not between
        row["review_interpretation"] = (
            "adjacent sampled sign change"
            if not between
            else "endpoint sign change only; interpolation unsupported across missing points"
        )
        reviewed_crossings.append(row)
    write_csv(out / "crossings_reviewed.csv", reviewed_crossings)
    sensitivity_crossings = []
    grouped_clocks = collections.defaultdict(list)
    for row in clocks:
        if float(row["ratio"]) > 0:
            grouped_clocks[clock_group(row)].append(row)
    for group in grouped_clocks.values():
        group.sort(key=lambda r: float(r["ratio"]))
        for a, b in zip(group, group[1:]):
            da, db = (a["finite_delta50"], b["finite_delta50"])
            if (
                a["energy_only_final_pass"]
                and b["energy_only_final_pass"]
                and (da is not None)
                and (db is not None)
                and (da * db < 0)
            ):
                x = math.log(float(a["ratio"])) + (math.log(float(b["ratio"])) - math.log(float(a["ratio"]))) * -da / (
                    db - da
                )
                sensitivity_crossings.append(
                    {
                        **{k: a[k] for k in ("dynamics", "recommender", "zeta", "epsilon", "scan", "b")},
                        "left_ratio": a["ratio"],
                        "right_ratio": b["ratio"],
                        "interpolation": math.exp(x),
                        "uses_posthoc_energy_check": a["energy_only_added"] or b["energy_only_added"],
                    }
                )
    write_csv(out / "energy_only_sensitivity_crossings.csv", sensitivity_crossings)
    summary["clock_primary_verified"] = sum((r["primary_joint_plateau"] for r in clocks))
    summary["clock_unresolved"] = sum((not r["primary_joint_plateau"] for r in clocks))
    summary["energy_only_posthoc_added"] = sum((r["energy_only_added"] for r in clocks))
    summary["clock_missing_by_rec"] = dict(
        collections.Counter((r["recommender"] for r in clocks if not r["primary_joint_plateau"]))
    )
    summary["clock_progress_inconsistent"] = sum((r["order_consistent"] == "False" for r in clocks))
    summary["old_new_disagreement_positive_q"] = sum(
        (r["old_new_order_disagree"] == "True" and float(r["ratio"]) > 0 for r in clocks)
    )
    summary["q0_spurious_disagreements"] = sum(
        (r["old_new_order_disagree"] == "True" and r["q0_roundoff_tie"] for r in clocks)
    )
    summary["clock_crossings_across_missing"] = sum((not r["adjacent_in_design"] for r in reviewed_crossings))
    summary["clock_budget_max_residual"] = max((abs(float(r["budget_residual"])) for r in clocks))
    summary["clock_positive_motion_cases"] = sum((float(r["budget_motion"]) > 1e-12 for r in clocks))
    summary["clock_positive_rewiring_cases"] = sum((float(r["budget_rewire"]) > 1e-12 for r in clocks))
    summary["clock_max_mass_residual"] = max((abs(float(r["mass_residual"])) for r in clocks))
    summary["clock_max_degree_residual"] = max((abs(float(r["row_residual"])) for r in clocks))
    summary["energy_only_added_tail_change_max"] = {
        c: max(
            (
                r[c + "_tail_relative_change"]
                for r in clocks
                if r["energy_only_added"] and r[c + "_tail_relative_change"] is not None
            ),
            default=None,
        )
        for c in ("rho", "edge")
    }


def clock_estimates(root, out):
    """Write T50-of-mean-curve interval/coverage rows for the declared clock ensembles."""
    rows = read_table(root / "analysis" / "micro" / "conditional_process_intervals.csv")
    selected = []
    for r in rows:
        if "clock_ensemble" not in r["roles"] or "T50_of_mean_energy_curve" not in r["metric"]:
            continue
        alpha = number(r, "alpha")
        a, lo, hi, pair = (alpha * number(r, k) for k in ("T50_of_mean_curve", "lower", "upper", "pair_T50"))
        selected.append(
            {
                "dynamics": r["dynamics"],
                "recommender": r["recommender"],
                "initial": int(r["initial_replicate"]),
                "channel": r["metric"].split("_")[0],
                "process_count": int(r["process_count"]),
                "micro_tau50": a,
                "lower": lo,
                "upper": hi,
                "pair_tau50": pair,
                "relative_error": abs(pair / a - 1),
                "pair_within_mean_interval": lo <= pair <= hi,
            }
        )
    table(out / "conditional_clock_prediction.csv", selected)
    return selected


def crossing_ranges(root, out):
    """Write local sampled sign-change ranges, preserving unconfirmed/missing crossing points."""
    rows = read_table(root / "analysis" / "clocks" / "crossing_intervals.csv")
    groups = collections.defaultdict(list)
    for r in rows:
        if r["criterion"] == "energy":
            groups[(r["dynamics"], int(float(r["zeta"])), int(r["p"]))].append(r)
    answer = []
    for (d, z, p), rs in sorted(groups.items()):
        brackets = sorted({(number(r, "left_ratio"), number(r, "right_ratio")) for r in rs})
        answer.append(
            {
                "dynamics": d,
                "zeta": z,
                "progress": p,
                "numeric_configurations": len(rs),
                "interpolation_min": min(number(r, "interpolation") for r in rs),
                "interpolation_max": max(number(r, "interpolation") for r in rs),
                "sample_brackets": brackets,
                "scope": "numerical setting range, not a confidence interval or a phase transition",
            }
        )
    table(out / "crossing_ranges.csv", answer)
    return answer
