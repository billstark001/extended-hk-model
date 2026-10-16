"""Long-window clocks with independent energy confirmation and coverage evidence."""

from __future__ import annotations

import collections
import math

from common.io.tables import table
from theory.release import clock_pair, confirmed_clocks

MICRO_KEYS = (
    "dynamics",
    "recommender",
    "zeta",
    "alpha",
    "ratio",
    "epsilon",
    "n",
    "b",
    "quadrature",
    "initial_replicate",
)
CLOCK_KEYS = ("dynamics", "recommender", "zeta", "alpha", "epsilon", "b", "quadrature", "role")


def clock_row(record, protocol, frames=None):
    """Reduce a long trace to scaled-time quantile clocks with independent energy plateau confirmation and budget diagnostics."""
    md, value = record["job"]["meta"], record["result"]
    trace = value if frames is None else {**value, "frames": frames}
    energy = confirmed_clocks(trace, md["energy_check"], protocol["clock_zero_tolerance"])
    joint = clock_pair(trace) if frames is None else None
    row = {
        **md,
        "id": record["job"]["id"],
        "actual_tau": md["alpha"] * trace["frames"][-1]["time"],
        **energy,
        "joint_confirmed": bool(value["diagnostics"]["converged"]) if frames is None else None,
        "joint_delta50": joint["delta50"] if joint else None,
    }
    for p in (10, 50, 90):
        for ch in ("rho", "edge"):
            t = row[f"{ch}_T{p}"]
            row[f"{ch}_tau{p}"] = md["alpha"] * t if t is not None else None
    last, first = trace["frames"][-1], trace["frames"][0]
    residual = last["edge_energy"] - first["edge_energy"] - last["budget_rewire"] - last["budget_motion"]
    if abs(residual) > 1e-10:
        raise ValueError("split energy budget does not close")
    row.update(budget_residual=residual, budget_rewire=last["budget_rewire"], budget_motion=last["budget_motion"])
    for ch in ("rho", "edge"):
        a, b = row[ch + "_T50"], row[ch + "_T50_tail20"]
        row[ch + "_tail_relative_change"] = abs(b / a - 1) if a and b else None
    return row


def analyze_ensembles(items, jobs, protocol, out):
    """Write long-horizon clock tables, local crossing ranges and missing/unconfirmed design evidence under the saved protocol."""
    rows = [clock_row(r, protocol) for r in items]
    table(out / "direct_clocks.csv", rows)
    prefixes = []
    for record in items:
        if record["job"]["meta"]["role"] != "extension":
            continue
        alpha = record["job"]["meta"]["alpha"]
        for tau in protocol["extension_prefix_taus"]:
            frames = [f for f in record["result"]["frames"] if alpha * f["time"] <= tau + 1e-9]
            if frames and math.isclose(alpha * frames[-1]["time"], tau, abs_tol=1e-8):
                prefixes.append(
                    {**clock_row(record, protocol, frames), "requested_prefix_tau": tau, "prefix_observed": True}
                )
            else:
                prefixes.append(
                    {
                        "id": record["job"]["id"],
                        "requested_prefix_tau": tau,
                        "prefix_observed": False,
                        "actual_final_tau": alpha * record["result"]["frames"][-1]["time"],
                        "joint_stop_before_prefix": record["result"]["diagnostics"]["converged"],
                    }
                )
    table(out / "extension_prefixes.csv", prefixes)
    by_id = {r["id"]: r for r in rows}
    strips = collections.defaultdict(list)
    for job in jobs:
        md = job["meta"]
        if md["role"] in ("crossing", "alpha_crossing"):
            strips[tuple(md[k] for k in CLOCK_KEYS)].append(job)
    crossings = []
    for group in strips.values():
        group.sort(key=lambda j: j["meta"]["ratio"])
        for left, right in zip(group, group[1:]):
            a, b = by_id.get(left["id"]), by_id.get(right["id"])
            if a is None or b is None:
                continue  # Failed/unrun design points break the strip.
            for criterion in ("energy", "joint"):
                if not a[criterion + "_confirmed"] or not b[criterion + "_confirmed"]:
                    continue
                for p in (10, 50, 90):
                    da, db = a[f"delta{p}"], b[f"delta{p}"]
                    if da is None or db is None or da == db or da * db > 0:
                        continue
                    x = math.log(a["ratio"]) + (math.log(b["ratio"]) - math.log(a["ratio"])) * (-da) / (db - da)
                    crossings.append(
                        {
                            **{k: a[k] for k in CLOCK_KEYS},
                            "criterion": criterion,
                            "p": p,
                            "left_ratio": a["ratio"],
                            "right_ratio": b["ratio"],
                            "left_delta": da,
                            "right_delta": db,
                            "interpolation": math.exp(x),
                            "interpretation": "adjacent predeclared samples; observable-specific sign crossing",
                        }
                    )
    table(out / "crossing_intervals.csv", crossings)
    return {
        "trajectories": len(rows),
        "joint_confirmed": sum(r["joint_confirmed"] for r in rows),
        "energy_confirmed": sum(r["energy_confirmed"] for r in rows),
        "adjacent_crossings": len(crossings),
    }
