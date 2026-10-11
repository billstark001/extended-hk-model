"""Paired release quantiles, plateau checks, budgets and sampled crossing intervals."""

from __future__ import annotations

import collections
import math

from common.io.tables import table
from theory.release import clock_pair
from theory.runtime.dataset import key

from ..plot import plot_clocks


def analyze_clocks(items, out):
    """Write direct paired quantile clocks, budget/plateau diagnostics and sampled crossing intervals from validated records."""
    rows = []
    for rec in items:
        meta = rec["job"]["meta"]
        trace = rec["result"]
        last = trace["frames"][-1]
        first = trace["frames"][0]
        budget_error = last["edge_energy"] - first["edge_energy"] - last["budget_rewire"] - last["budget_motion"]
        if abs(budget_error) > 1e-10:
            raise ValueError("split energy budget fails")
        row = {
            **meta,
            "id": rec["job"]["id"],
            **clock_pair(trace),
            "pathway": trace["pathway"],
            "edge_minus_marginal_reference": last["edge_energy"] - last["reference_energy"],
            "budget_rewire": last["budget_rewire"],
            "budget_motion": last["budget_motion"],
            "budget_residual": budget_error,
            "final_polarization": trace.get("polarization", [None])[-1],
            "final_homophily": trace.get("homophily", [None])[-1],
            "mass_residual": trace["diagnostics"]["max_node_mass_residual"],
            "row_residual": trace["diagnostics"]["max_fixed_degree_residual"],
        }
        row["old_new_order_disagree"] = (
            (row["pathway"] > 0.6) != (row["delta50"] < 0)
            if row["delta50"] is not None and row["pathway"] is not None
            else None
        )
        for channel in ("rho", "edge"):
            for p in (10, 50, 90):
                t = row[f"{channel}_T{p}"]
                row[f"{channel}_tau{p}"] = meta["alpha"] * t if t is not None else None
        rows.append(row)
    table(out / "direct_clocks.csv", rows)
    sensitivity = [
        {
            **{k: r[k] for k in ("id", "dynamics", "recommender", "zeta", "epsilon", "alpha", "ratio", "b")},
            "order_consistent": r["order_consistent"],
            "rho_status": r["rho_status"],
            "edge_status": r["edge_status"],
            **{k: r[k] for k in ("delta10", "delta50", "delta90", "rho_T50_tail20", "edge_T50_tail20")},
        }
        for r in rows
    ]
    table(out / "clock_sensitivity.csv", sensitivity)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key({k: r[k] for k in ("dynamics", "recommender", "zeta", "epsilon", "alpha", "b", "scan")})].append(r)
    crossings = []
    for group in groups.values():
        for p in (10, 50, 90):
            valid = sorted(
                [r for r in group if r["ratio"] > 0 and r[f"delta{p}"] is not None], key=lambda r: r["ratio"]
            )
            for a, b in zip(valid, valid[1:]):
                da, db = a[f"delta{p}"], b[f"delta{p}"]
                if da * db <= 0 and da != db:
                    x = math.log(a["ratio"]) + (math.log(b["ratio"]) - math.log(a["ratio"])) * (-da) / (db - da)
                    crossings.append(
                        {
                            **{k: a[k] for k in ("dynamics", "recommender", "zeta", "epsilon", "alpha", "b", "scan")},
                            "p": p,
                            "crossing_ratio": math.exp(x),
                            "left_ratio": a["ratio"],
                            "right_ratio": b["ratio"],
                            "interpretation": "linear in log ratio; no phase-transition claim",
                        }
                    )
    table(out / "crossing_intervals.csv", crossings)
    table(
        out / "unresolved_clocks.csv",
        [
            {
                k: v
                for k, v in r.items()
                if k
                in (
                    "id",
                    "dynamics",
                    "recommender",
                    "zeta",
                    "ratio",
                    "epsilon",
                    "alpha",
                    "b",
                    "horizon",
                    "rho_status",
                    "edge_status",
                    "rho_nonmonotone_release",
                    "edge_nonmonotone_release",
                )
            }
            for r in rows
            if r["delta50"] is None
        ],
    )
    return {
        "curves": len(rows),
        "verified_plateaus": sum(r["rho_plateau_verified"] for r in rows),
        "unresolved_plateaus": sum(not r["rho_plateau_verified"] for r in rows),
        "figures": plot_clocks(items, rows, out),
    }
