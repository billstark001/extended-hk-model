"""Estimate task cost using a supplied completed baseline; never simulate."""

import collections
import json
import math
from pathlib import Path

import numpy as np

from common.io.json import read_result


def estimate(jobs, baseline, workers):
    """Return a heuristic wall-time estimate from completed baseline timings; report missing comparables explicitly and never run a solver."""
    # TODO(cost-schema): this legacy jobs/meta/payload adapter is study-specific.
    # Move schema translation to its paper owner before sharing the pure cost
    # aggregation with the task-based theory/lifted workflows; preserve old fits.
    plan_path, status_path = baseline / "plan.json", baseline / "status.json"
    if not plan_path.exists() or not status_path.exists():
        return {"available": False, "reason": "completed baseline timings unavailable", "jobs": len(jobs)}
    plan = json.loads(plan_path.read_text())
    status = json.loads(status_path.read_text())
    times = {r["id"]: r["seconds"] for r in status["records"] if r["status"] == "complete"}
    for record in status["records"]:
        if record["status"] == "cached" and record.get("path") and Path(record["path"]).exists():
            times[record["id"]] = read_result(record["path"])["seconds"]
    candidates = [
        j for j in plan["jobs"] if j["id"] in times and (j["family"] != "micro" or j["payload"]["branches"] == 0)
    ]
    seconds = collections.defaultdict(float)
    missing = []
    for job in jobs:
        md, req = job["meta"], job["payload"]["request"]
        same = [
            j
            for j in candidates
            if j["family"] == job["family"] and all(j["meta"][k] == md[k] for k in ("dynamics", "recommender", "zeta"))
        ]
        if not same:
            missing.append(job["id"])
            continue

        def distance(old):
            om = old["meta"]
            return (
                4 * abs(math.log((om["ratio"] + 0.05) / (md["ratio"] + 0.05)))
                + 5 * abs(om["epsilon"] - md["epsilon"])
                + abs(math.log(om["b"] / md["b"]))
            )

        best = min(distance(j) for j in same)
        close = [j for j in same if distance(j) <= best + 1e-9]
        ref = close[0]
        om, old_req = ref["meta"], ref["payload"]["request"]
        elapsed = float(np.median([times[j["id"]] for j in close]))
        old_tau = om["alpha"] * old_req["dt"] * old_req["steps"]
        new_tau = md["alpha"] * req["dt"] * req["steps"]
        quad_ratio = req["resolution"]["opinion_quadrature_points"] / old_req["resolution"]["opinion_quadrature_points"]
        if job["family"] == "micro":
            # N expansion here is Random only. Mix graph-step work with pair
            # matrix/kernel work; this is an estimate, not a complexity bound.
            scale = 0.75 * md["n"] / om["n"] + 0.25 * (md["b"] / om["b"]) ** 2 * quad_ratio
        else:
            scale = (md["b"] / om["b"]) ** 2 * (1 + (0.4 if md["dynamics"] == "hk" else 0.15) * (quad_ratio - 1))
        scale *= new_tau / old_tau * om["alpha"] / md["alpha"]
        seconds[(job["family"], md.get("role", md.get("scan", job["tier"])))] += elapsed * scale
    serial = sum(seconds.values())
    central = serial / workers * 1.1
    return {
        "available": not missing,
        "unestimated_ids": missing,
        "baseline": str(baseline),
        "jobs": len(jobs),
        "workers": workers,
        "estimated_solver_wall_minutes": central / 60,
        "estimated_total_minutes_range": [central * 0.65 / 60 + 3, central * 1.8 / 60 + 10],
        "serial_seconds_by_role": {f"{f}/{r}": round(s, 1) for (f, r), s in sorted(seconds.items())},
        "assumptions": "same machine/backend; process-level parallelism; baseline median timings, horizon and N/mesh/quadrature extrapolation; stopping and I/O may change runtime; no wall deadline",
    }
