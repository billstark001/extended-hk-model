"""Within-initial-state process resampling and resolution-control comparisons."""

from __future__ import annotations

import collections

import numpy as np

from common.io.json import save_result
from common.io.tables import table
from common.statistics import bootstrap_weights, mean_interval
from theory.release import clocks, curve_clock_interval, release_clock
from theory.release.ensemble import clock_statistics
from theory.validation import compare_fields, trajectory_gap

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


def finite_micro_row(record, protocol):
    """Reduce one matched micro/pair record to physical-time clock and finite-population error fields using the saved protocol."""
    md, value = record["job"]["meta"], record["result"]
    micro, pair = value["frames"], value["pair"]["frames"]
    continuous = [{**f, "node_energy": f["exact_node_energy"], "edge_energy": f["exact_edge_energy"]} for f in micro]
    row = {
        **md,
        "id": record["job"]["id"],
        "initial_hash": value["initial_hash"],
        "projection_l1": value["projection_l1"],
        **compare_fields(micro[-1], pair[-1]),
        "final_micro_node_energy": continuous[-1]["node_energy"],
        "final_pair_node_energy": pair[-1]["node_energy"],
        "final_micro_binned_variance": micro[-1]["variance"],
        "finite_tau": md["alpha"] * micro[-1]["time"],
    }
    for channel in ("node", "edge"):
        name = channel + "_energy"
        row[channel + "_energy_gap"] = trajectory_gap(continuous, pair, name)["gap"]
        mc = clocks(continuous, False, name, protocol["tail_fractions"][0])
        pc = clocks(pair, False, name, protocol["tail_fractions"][0])
        for p in (10, 50, 90):
            row[f"{channel}_micro_T{p}"] = mc[f"T{p}"]
            row[f"{channel}_pair_T{p}"] = pc[f"T{p}"]
            row[f"{channel}_micro_tau{p}"] = md["alpha"] * mc[f"T{p}"] if mc[f"T{p}"] is not None else None
            row[f"{channel}_pair_tau{p}"] = md["alpha"] * pc[f"T{p}"] if pc[f"T{p}"] is not None else None
        row[channel + "_T50_relative_error"] = abs(pc["T50"] / mc["T50"] - 1) if pc["T50"] and mc["T50"] else None
        row[channel + "_micro_nonmonotone"] = mc["nonmonotone_release"]
        row[channel + "_bin_T50"] = clocks(micro, False, name, protocol["tail_fractions"][0])["T50"]
    return row


def cohorts(items, rows, protocol, out):
    """Group process branches by compatible initial-state and physical/numerical metadata for conditional resampling."""
    grouped = collections.defaultdict(list)
    for record in items:
        md = record["job"]["meta"]
        grouped[tuple(md[k] for k in MICRO_KEYS) + (md["alpha"] * record["result"]["frames"][-1]["time"],)].append(
            record
        )
    by_id = {r["id"]: r for r in rows}
    intervals, ensembles, curves, mean_rows = [], [], [], []
    for group, members in sorted(grouped.items()):
        md = members[0]["job"]["meta"]
        hashes = {r["result"]["initial_hash"] for r in members}
        seeds = [r["job"]["payload"]["process_seed"] for r in members]
        if len(hashes) != 1 or len(set(seeds)) != len(seeds):
            raise ValueError("cohort does not have one fixed initial state and unique process seeds")
        first = members[0]["result"]
        pair = first["pair"]["frames"]
        for r in members[1:]:
            other = r["result"]["pair"]["frames"]
            if (
                len(other) != len(pair)
                or any(
                    max(abs(a[k] - b[k]) for k in ("time", "node_energy", "edge_energy")) > 1e-12
                    for a, b in zip(pair, other)
                )
                or compare_fields(pair[-1], other[-1])["node_w1"] > 1e-12
            ):
                raise ValueError("deterministic pair trajectory changed with process seed")
        summary = {
            **dict(zip(MICRO_KEYS, group[:-1])),
            "finite_tau": group[-1],
            "initial_hash": next(iter(hashes)),
            "process_count": len(members),
            "roles": sorted({r["job"]["meta"]["role"] for r in members}),
            "uncertainty": "conditional on this initial graph/opinion state",
        }
        weights = bootstrap_weights(len(members), protocol)
        for metric in (
            "node_w1",
            "node_hellinger",
            "edge_hellinger",
            "node_energy_gap",
            "edge_energy_gap",
            "node_T50_relative_error",
            "edge_T50_relative_error",
            "final_micro_node_energy",
        ):
            values = [by_id[r["job"]["id"]][metric] for r in members]
            stat = mean_interval([np.nan if v is None else v for v in values], weights, protocol["confidence_level"])
            intervals.append({**summary, "metric": metric, **stat})
        times = np.asarray([f["time"] for f in first["frames"]])
        if any(not np.array_equal(times, [f["time"] for f in r["result"]["frames"]]) for r in members):
            raise ValueError("unequal micro horizons or record times in one cohort")
        curve_record = {**summary, "times": times.tolist(), "alpha": md["alpha"]}
        for channel in ("node", "edge"):
            matrix = np.asarray([[f["exact_" + channel + "_energy"] for f in r["result"]["frames"]] for r in members])
            ci, mean, band = curve_clock_interval(times, matrix, weights, protocol)
            pair_clock = clocks(pair, False, channel + "_energy", protocol["tail_fractions"][0])["T50"]
            intervals.append({**summary, "metric": channel + "_T50_of_mean_energy_curve", **ci, "pair_T50": pair_clock})
            curve_record[channel + "_mean"] = mean.tolist()
            curve_record[channel + "_lower"] = band[0].tolist()
            curve_record[channel + "_upper"] = band[1].tolist()
            curve_record[channel + "_pair_times"] = [f["time"] for f in pair]
            curve_record[channel + "_pair"] = [f[channel + "_energy"] for f in pair]
        matrices = {
            channel: np.asarray([[f["exact_" + channel + "_energy"] for f in r["result"]["frames"]] for r in members])
            for channel in ("node", "edge")
        }
        tau = times * md["alpha"]
        pair_times = np.asarray([f["time"] for f in pair]) * md["alpha"]
        pair_clocks = {ch: release_clock(pair_times, [f[ch + "_energy"] for f in pair]) for ch in ("node", "edge")}
        for plateaus in (None, {ch: c.plateau for ch, c in pair_clocks.items()}):
            stat, _ = clock_statistics(
                tau,
                matrices["node"],
                matrices["edge"],
                weights,
                confidence=protocol["confidence_level"],
                plateaus=plateaus,
            )
            mean_rows.append(
                {
                    **summary,
                    **stat,
                    "time_units": "tau=alpha*t",
                    **{"pair_" + ch + "_mean": c.area for ch, c in pair_clocks.items()},
                }
            )
        curves.append(curve_record)
        average = {
            name: np.asarray([r["result"]["frames"][-1][name] for r in members]).mean(axis=0).tolist()
            for name in ("rho", "edge")
        }
        discrepancy = compare_fields(average, pair[-1])
        boot_rho = weights @ np.asarray([r["result"]["frames"][-1]["rho"] for r in members])
        reference_rho = np.asarray(pair[-1]["rho"])
        reference_rho /= reference_rho.sum()
        boot_w1 = (
            np.abs(np.cumsum(boot_rho / boot_rho.sum(axis=1)[:, None] - reference_rho, axis=1)).sum(axis=1)
            * 2
            / md["b"]
        )
        tail = (1 - protocol["confidence_level"]) / 2
        lower, upper = np.quantile(boot_w1, (tail, 1 - tail))
        ensembles.append(
            {
                **summary,
                **discrepancy,
                "axis": first["pair"]["axis"],
                "node_w1_lower": float(lower) if len(members) > 1 else None,
                "node_w1_upper": float(upper) if len(members) > 1 else None,
                "micro_mean_rho": average["rho"],
                "micro_mean_edge": average["edge"],
                "pair_rho": pair[-1]["rho"],
                "pair_edge": pair[-1]["edge"],
            }
        )
    table(out / "conditional_mean_release.csv", mean_rows)
    table(out / "conditional_process_intervals.csv", intervals)
    table(
        out / "ensemble_distribution_errors.csv",
        [
            {
                k: v
                for k, v in r.items()
                if k not in ("axis", "micro_mean_rho", "micro_mean_edge", "pair_rho", "pair_edge")
            }
            for r in ensembles
        ],
    )
    save_result(out / "ensemble_distributions.json.gz", ensembles)
    save_result(out / "conditional_energy_curves.json.gz", curves)
    # Equal four-process comparisons for N=500/1000/2000; larger-N graphs are
    # new initial realizations, not copies of the N=500 graph.
    population = []
    for r in rows:
        if (
            r["dynamics"] == "hk"
            and r["recommender"] == "random"
            and r["alpha"] == 0.02
            and r["ratio"] == 10
            and r["b"] == 161
            and r["quadrature"] == 33
            and r["process_replicate"] < 4
            and r["finite_tau"] == 30
        ):
            population.append(r)
    table(out / "population_four_processes.csv", population)
    return curves, len(grouped)


def analyze_ensembles(items, protocol, out):
    """Write branch, conditional interval and resolution-control tables from matched records; initial states remain separate cohorts."""
    rows = [finite_micro_row(r, protocol) for r in items]
    # Mesh and quadrature cannot alter the underlying micro process.
    graph_hashes = collections.defaultdict(set)
    matched_processes = collections.defaultdict(list)
    for r in items:
        md = r["job"]["meta"]
        graph_hashes[(md["init_seed"], md["n"])].add(r["result"]["initial_hash"])
        matched_processes[
            tuple(md[k] for k in MICRO_KEYS if k not in ("b", "quadrature"))
            + (md["process_seed"], md["alpha"] * r["result"]["frames"][-1]["time"])
        ].append(r)
    if any(len(h) != 1 for h in graph_hashes.values()):
        raise ValueError("initial graph/opinions changed under a mesh/recommender/process comparison")
    for members in matched_processes.values():
        reference = members[0]["result"]["frames"]
        for record in members[1:]:
            other = record["result"]["frames"]
            if len(reference) != len(other) or any(
                max(abs(a[k] - b[k]) for k in ("time", "exact_node_energy", "exact_edge_energy")) > 1e-12
                for a, b in zip(reference, other)
            ):
                raise ValueError("mesh/quadrature changed the underlying micro process")
    table(out / "trajectory_errors.csv", rows)
    curves, count = cohorts(items, rows, protocol, out)
    return {
        "trajectories": len(rows),
        "fixed_initial_cohorts": count,
        "initial_hash_checks": "passed",
        "pair_process_invariance": "passed",
        "micro_mesh_quadrature_invariance": "passed",
    }
