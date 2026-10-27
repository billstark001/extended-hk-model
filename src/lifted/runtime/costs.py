"""Pure task-cost estimates from authenticated pilot records, with explicit extrapolation."""

import math

import numpy as np


def task_cost(task, measurements):
    """Predict serial seconds; full-horizon censored pilots scale by requested steps.

    Timings include classification and snapshot I/O. The mean completed path
    length is descriptive, not an assertion about the unsampled tail. A hard
    execution deadline bounds work independently of these heuristic estimates.
    """
    kind = task["kind"]
    if kind.endswith("extend"):
        kind = kind.replace("extend", "probe")
    samples = [r for r in measurements if r["kind"] == kind]
    if not samples:
        raise ValueError(f"pilot has no cost measurements for {kind}")
    p = task["parameters"]

    def distance(row):
        return (
            10 * (row["variant"] != task["variant"])
            + 4 * abs(math.log((row["ratio"] + 0.05) / (p["ratio"] + 0.05)))
            + 2 * abs(math.log(row["bins"] / p["bins"]))
            + 2 * abs(math.log(row["population"] / p["population"]))
            + 2 * abs(math.log1p(row.get("source_step", 0) / 50) - math.log1p(task["source_step"] / 50))
        )

    nearest = min(distance(r) for r in samples)
    chosen = [r for r in samples if abs(distance(r) - nearest) < 1e-10]
    estimates = []
    for row in chosen:
        overhead = max(0.01, row["wall_seconds"] - row["elapsed_seconds"])
        if kind == "initial":
            estimates.append(row["wall_seconds"] * p["population"] / row["population"])
            continue
        per_step = row["elapsed_seconds"] / max(1, row["executed_steps"])
        # Micro work is nearly linear in N; dense coarse work is approximately
        # quadratic/cubic in bins. Exact grid measurements take precedence.
        scaling = p["population"] / row["population"] if kind.startswith("micro_") else (p["bins"] / row["bins"]) ** 2.5
        horizon = task["horizon_steps"]
        mean_steps = row["executed_steps"] / row["paths"]
        if kind in ("micro_probe", "micro_tail") or (kind == "meso_probe" and task["variant"] == "stochastic"):
            if row["unresolved"]:
                ratio = min(1, row["unresolved"] / row["paths"])
                mean_steps = mean_steps + ratio * max(0, horizon - row["horizon_steps"])
            steps = min(horizon, max(1, mean_steps)) * len(task["replicates"])
        else:
            steps = horizon * len(task["replicates"])
        estimates.append(overhead + per_step * scaling * steps)
    return float(np.median(estimates)), nearest > 1e-10


def plan_cost(tasks, measurements, *, workers, reserve_seconds, cache=None):
    if workers < 1:
        raise ValueError("positive worker count required")
    by_kind: dict[str, float] = {}
    extrapolated: list[str] = []
    cache = {} if cache is None else cache
    for task in tasks:
        if task["id"] not in cache:
            cache[task["id"]] = task_cost(task, measurements)
        cost, extrapolate = cache[task["id"]]
        by_kind[task["kind"]] = by_kind.get(task["kind"], 0) + cost
        if extrapolate:
            extrapolated.append(task["id"])
    compute = sum(by_kind.values()) / workers * 1.15
    return dict(
        expected_hours=(compute + reserve_seconds) / 3600,
        solver_wall_seconds=compute,
        serial_seconds_by_kind=by_kind,
        extrapolated_tasks=len(extrapolated),
        assumptions="same hardware/backend; pilot mean path length; 15% scheduling overhead; tail length uncertain",
    )
