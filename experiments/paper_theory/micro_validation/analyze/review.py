"""Continuous microscopic energy and conditional-bias evidence for manuscript reports."""

from __future__ import annotations

import numpy as np

from experiments.paper_theory.reporting import raw, summarize, write_csv
from theory.release import clocks as energy_clocks
from theory.validation import trajectory_gap


def review_micro(root, out, summary, micro):
    """Add continuous-energy/finite-clock/conditional-bias evidence to summary and write report tables; return the selected basin record if present."""
    detail, audit_rows, micro_clocks, selected = ([], [], [], None)
    for row in micro:
        record = raw(root, "micro", row["id"])
        value = record["result"]
        a = [
            {**f, "node_energy": f["exact_node_energy"], "edge_energy": f["exact_edge_energy"]} for f in value["frames"]
        ]
        final = value["frames"][-1]
        detail.append(
            {
                "id": row["id"],
                "tier": row["tier"],
                "dynamics": row["dynamics"],
                "recommender": row["recommender"],
                "zeta": row["zeta"],
                "ratio": row["ratio"],
                "replicate": row["replicate"],
                "exact_node_gap": trajectory_gap(a, value["pair"]["frames"], "node_energy")["gap"],
                "exact_edge_gap": trajectory_gap(a, value["pair"]["frames"], "edge_energy")["gap"],
                "micro_final_variance": final["variance"],
                "micro_final_node_energy": final["exact_node_energy"],
                "pair_final_node_energy": value["pair"]["frames"][-1]["node_energy"],
                "max_node_bin_bias": max((abs(f["node_energy"] - f["exact_node_energy"]) for f in value["frames"])),
                "max_edge_bin_bias": max((abs(f["edge_energy"] - f["exact_edge_energy"]) for f in value["frames"])),
            }
        )
        clock_row = {
            k: row[k]
            for k in ("id", "tier", "dynamics", "recommender", "zeta", "ratio", "replicate", "law", "alpha", "b")
        }
        clock_row["interpretation"] = (
            "finite common horizon and own last-10-percent energy reference; no asymptotic plateau verification"
        )
        for channel in ("node", "edge"):
            mc = energy_clocks(value["frames"], False, channel + "_energy")
            pc = energy_clocks(value["pair"]["frames"], False, channel + "_energy")
            for name in ("T10", "T50", "T90", "amplitude", "nonmonotone_release"):
                clock_row[channel + "_micro_" + name] = mc[name]
                clock_row[channel + "_pair_" + name] = pc[name]
            clock_row[channel + "_T50_relative_error"] = (
                abs(pc["T50"] / mc["T50"] - 1) if pc["T50"] and mc["T50"] else None
            )
        micro_clocks.append(clock_row)
        for audit in value.get("audits") or []:
            counts = np.asarray(audit["source_counts"])
            weights = counts / counts.sum()
            actual = np.asarray(audit["mean_increment"])
            pred = float(row["alpha"]) * np.asarray(audit["frozen"]["mean"])
            actual2 = np.asarray(audit["second_increment"])
            pred2 = float(row["alpha"]) ** 2 * np.asarray(
                audit["frozen"]["hk_second" if row["dynamics"] == "hk" else "deffuant_second"]
            )
            signal = float(np.sqrt(weights @ pred**2))
            bias = float(np.sqrt(weights @ (actual - pred) ** 2))
            audit_rows.append(
                {
                    "id": row["id"],
                    "tier": row["tier"],
                    "dynamics": row["dynamics"],
                    "recommender": row["recommender"],
                    "zeta": row["zeta"],
                    "ratio": row["ratio"],
                    "tau": float(row["alpha"]) * audit["step"],
                    "weighted_mean_bias": bias,
                    "weighted_mean_signal": signal,
                    "relative_mean_bias": bias / signal if signal > 1e-15 else None,
                    "weighted_second_bias": float(np.sqrt(weights @ (actual2 - pred2) ** 2)),
                }
            )
        if (
            row["tier"] == "core"
            and row["dynamics"] == "hk"
            and (row["recommender"] == "random")
            and (float(row["ratio"]) == 10)
            and (row["replicate"] == "0")
        ):
            selected = record
    write_csv(out / "micro_continuous_energy.csv", detail)
    write_csv(out / "micro_finite_time_clocks.csv", micro_clocks)
    summary["micro_finite_clocks_by_ratio"] = summarize(
        [r for r in micro_clocks if r["tier"] == "core"],
        ("ratio",),
        ("node_T50_relative_error", "edge_T50_relative_error"),
    )
    write_csv(out / "conditional_weighted_bias.csv", audit_rows)
    summary["micro_grid_projection_max_l1"] = max((float(r["projection_l1"]) for r in micro))
    summary["basin_example_id"] = selected["job"]["id"] if selected else None
    summary["hk_random_fast_cases"] = [
        r
        for r in detail
        if r["tier"] == "core"
        and r["dynamics"] == "hk"
        and (r["recommender"] == "random")
        and (float(r["ratio"]) == 10)
    ]
    summary["conditional_weighted"] = summarize(
        audit_rows, ("tier", "tau"), ("weighted_mean_bias", "relative_mean_bias", "weighted_second_bias")
    )
    return selected
