"""Finite-population micro/pair errors with initial-state-level statistical aggregation."""

import json
from collections import defaultdict

import numpy as np

from theory.scaling import curve_distance

from ...data import write_csv


def analyze(directory):
    """Write branch metrics and population summaries; average process branches within each initial state before estimating between-initial SEM."""
    rows = []
    for path in sorted((directory / "raw").glob("*.json")):
        value = json.loads(path.read_text())
        task, result = value["task"], value["result"]
        p = task["parameters"]
        micro, pair = result["frames"], result["pair"]["frames"]
        mt = np.array([f["time"] for f in micro]) * p["alpha"]
        pt = np.array([f["time"] for f in pair]) * p["alpha"]
        final = next(f for f in reversed(pair) if f.get("rho"))
        a, b = np.array(micro[-1]["rho"]), np.array(final["rho"])
        if not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1):
            raise ValueError(f"invalid final histogram: {path}")
        row = {
            "id": task["id"],
            "dynamics": p["dynamics"],
            "recommender": p["recommender"],
            "r": p["ratio"],
            "n": p["population"],
            "initial": task["initial"],
            "branch": task["branch"],
            "horizon_tau": min(mt[-1], pt[-1]),
            "micro_variance": micro[-1]["exact_variance"],
            "micro_discordant": micro[-1]["exact_discordant"],
            "pair_variance": pair[-1]["variance"],
            "pair_discordant": pair[-1]["discordant"],
            "final_node_w1": float(np.sum(abs(np.cumsum(a - b))) * 2 / p["bins"]),
        }
        for name in ("node_energy", "edge_energy"):
            distance = curve_distance(pt, [f[name] for f in pair], mt, [f["exact_" + name] for f in micro])
            row[f"{name}_gap_max"] = distance["absolute_max"]
            row[f"{name}_gap_relative"] = distance["relative_max"]
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no microscopic results; execute a design first")
    write_csv(directory / "tables" / "replicates.csv", rows)
    # Average process branches within each initial graph first. The independent
    # statistical units are graph/opinion initializations, not all branches.
    clusters = defaultdict(list)
    for row in rows:
        clusters[(row["dynamics"], row["recommender"], row["r"], row["n"], row["initial"])].append(row)
    metrics = (
        "micro_variance",
        "micro_discordant",
        "pair_variance",
        "pair_discordant",
        "final_node_w1",
        "node_energy_gap_max",
        "edge_energy_gap_max",
    )
    groups = defaultdict(list)
    for key, members in clusters.items():
        groups[key[:-1]].append(
            {**{m: float(np.mean([r[m] for r in members])) for m in metrics}, "branches": len(members)}
        )
    aggregate = []
    for key, initials in groups.items():
        row = dict(zip(("dynamics", "recommender", "r", "n"), key))
        row.update(initials=len(initials), branches=sum(i["branches"] for i in initials))
        for metric in metrics:
            values = np.array([i[metric] for i in initials])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        aggregate.append(row)
    write_csv(directory / "tables" / "population_summary.csv", aggregate)
    print(f"analyzed {len(rows)} branches, {len(clusters)} initial-state groups within conditions")
