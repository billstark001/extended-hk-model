"""Offline assembly from this design's frozen records, without historical inputs."""

import json
from collections import defaultdict

import numpy as np

from common.io.json import atomic_json, fingerprint, sha
from common.io.records import content_fingerprint, record_path
from common.statistics import bootstrap_weights, clustered_interval, hierarchical_weights, paired_difference
from theory.release import release_clock
from theory.release.ensemble import clock_statistics, plateau_check
from theory.runtime.records import read_micro
from theory.scaling import curve_distance
from theory.validation import compare_fields

from ..data import write_csv
from ..runtime.measure_analysis import cached_cases
from ..scaling.analyze import scaling_rows

KEYS = (
    "dynamics",
    "recommender",
    "steepness",
    "epsilon",
    "ratio",
    "alpha",
    "population",
    "bins",
    "quadrature",
    "dt",
    "horizon",
)


def key(p, exclude=()):
    return tuple(p[k] for k in KEYS if k not in exclude)


def metadata(p):
    return {k: p[k] for k in KEYS}


def compact_record(record):
    """Keep full fields only at snapshots/endpoints, not at every energy sample."""
    result = record["result"]
    result["pair_frames_fingerprint"] = content_fingerprint(result["pair"]["frames"])
    for name, fields in (
        ("frames", ("exact_node_energy", "exact_edge_energy")),
        ("pair", ("node_energy", "edge_energy")),
    ):
        frames = result[name] if name == "frames" else result[name]["frames"]
        for i, frame in enumerate(frames):
            if np.asarray(frame.get("edge", [])).size or i == len(frames) - 1:
                for field in ("rho", "edge"):
                    if field in frame:
                        frame[field] = np.asarray(frame[field], dtype=float)
            else:
                frames[i] = {k: frame[k] for k in ("time", "step", *fields)}
    return record


def pairing_excess(frame, epsilon):
    """Binned concordance relative to independent pairing at actual marginals."""
    edge = np.asarray(frame["edge"], dtype=float).reshape(len(frame["rho"]), -1)
    edge = edge / edge.sum()
    centers = -1 + (np.arange(len(edge)) + 0.5) * 2 / len(edge)
    concordance = np.abs(centers[:, None] - centers[None, :]) <= epsilon
    reference = np.outer(edge.sum(1), edge.sum(0))
    return float(np.sum((edge - reference) * concordance))


def check(time, values, protocol):
    return plateau_check(
        time,
        values,
        window=protocol["plateau_window_tau"],
        minimum_horizon=protocol["minimum_plateau_tau"],
        tolerance=protocol["plateau_relative_tolerance"],
    )


def clock_rows(time, energies, protocol, md):
    """Use the same long endpoint reference at every prefix, retaining tail sensitivity."""
    rows = []
    observed = float(time[-1])
    horizons = list(protocol["prefix_taus"])
    if protocol.get("adaptive_measure_stopping") and not any(abs(h - observed) < 1e-9 for h in horizons):
        horizons.append(observed)
    unobserved = ",".join(str(h) for h in protocol["prefix_taus"] if h > observed + 1e-9)
    for name, values in energies.items():
        long = release_clock(time, values)
        alternate = release_clock(time, values, plateau=float(np.median(values[-max(1, len(values) // 5) :])))
        for horizon in sorted(horizons):
            if horizon > time[-1] + 1e-9:
                continue
            grid = np.r_[time[time < horizon], horizon]
            clock = release_clock(grid, np.interp(grid, time, values), plateau=long.plateau)
            rows.append(
                {
                    **md,
                    "channel": name,
                    "horizon_tau": horizon,
                    "mean_tau": clock.area,
                    "t50_tau": clock.t50,
                    "plateau": long.plateau,
                    "amplitude": long.amplitude,
                    "reference_horizon_tau": float(time[-1]),
                    "is_observed_endpoint": abs(horizon - observed) < 1e-9,
                    "unobserved_prefix_taus": unobserved,
                    "plateau_confirmed": check(time, values, protocol),
                    "status": long.status,
                    "rebound": clock.rebound,
                    "negative_area": clock.negative_area,
                    "tail20_mean_tau": alternate.area,
                    "tail20_relative_change": abs(alternate.area / long.area - 1) if long.area > 0 else np.nan,
                    "fraction_of_long_area": clock.area / long.area if long.area > 0 else np.nan,
                }
            )
    return rows


def measure_tables(cases, tasks, protocol):
    clocks, comparisons, snapshots = [], [], []
    for case in cases:
        task = tasks[case["source"].removesuffix(".npz")]
        clocks.extend(
            clock_rows(
                case["tau"],
                {n: case[n + "_energy"] for n in ("node", "edge")},
                protocol,
                {
                    **metadata(case),
                    "id": task["id"],
                    "roles": ",".join(task["roles"]),
                    "planned_maximum_tau": task["parameters"]["horizon"],
                    "stopped_early": bool(case["diagnostics"].get("converged", False)),
                    "stop_reason": case["diagnostics"].get("stop_reason", ""),
                },
            )
        )
    # Fixed-r alpha collapse uses raw energies; no endpoint alignment.
    selected = [c for c in cases if "scaling" in tasks[c["source"].removesuffix(".npz")]["roles"]]
    comparisons = scaling_rows(selected)
    groups = defaultdict(list)
    for c in selected:
        groups[key(c, ("alpha",))].append(c)
    for members in groups.values():
        ref = min(members, key=lambda c: c["alpha"])
        for c in members:
            if c is ref:
                continue
            rt = np.asarray(ref["snapshot_time"]) * ref["alpha"]
            ct = np.asarray(c["snapshot_time"]) * c["alpha"]
            for i, tau in enumerate(ct):
                hits = np.flatnonzero(np.isclose(rt, tau, atol=1e-9, rtol=0))
                if len(hits) and tau > 0:
                    j = hits[0]
                    snapshots.append(
                        {
                            **metadata(c),
                            "reference_alpha": ref["alpha"],
                            "tau": float(tau),
                            **compare_fields(
                                {"rho": c["snapshot_rho"][i], "edge": c["snapshot_edge"][i]},
                                {"rho": ref["snapshot_rho"][j], "edge": ref["snapshot_edge"][j]},
                            ),
                        }
                    )
    # Single-factor numerical checks; do not mix grid and alpha controls.
    numerics = []
    selected = [c for c in cases if "numerics" in tasks[c["source"].removesuffix(".npz")]["roles"]]
    groups = defaultdict(list)
    for c in selected:
        groups[key(c, ("bins", "quadrature", "dt"))].append(c)
    for members in groups.values():
        base = next(
            (
                c
                for c in members
                if c["dt"] == 1
                and c["bins"]
                == protocol.get("numerical_baseline", {}).get("bins", 21 if protocol["prefix_taus"][-1] < 1 else 161)
                and c["quadrature"]
                == protocol.get("numerical_baseline", {}).get("quadrature", 3 if protocol["prefix_taus"][-1] < 1 else 9)
            ),
            None,
        )
        if base is None:
            continue
        for c in members:
            if c is base:
                continue
            for n in ("node", "edge"):
                a, b = release_clock(base["tau"], base[n + "_energy"]), release_clock(c["tau"], c[n + "_energy"])
                numerics.append(
                    {
                        **metadata(c),
                        "channel": n,
                        "reference_bins": base["bins"],
                        "reference_quadrature": base["quadrature"],
                        "reference_dt": base["dt"],
                        "reference_observed_horizon_tau": float(base["tau"][-1]),
                        "observed_horizon_tau": float(c["tau"][-1]),
                        "final_fields_at_own_observed_endpoints": True,
                        "mean_relative_change": abs(b.area / a.area - 1) if a.area > 0 else np.nan,
                        **curve_distance(base["tau"], base[n + "_energy"], c["tau"], c[n + "_energy"]),
                        **compare_fields(
                            {"rho": base["final_rho"], "edge": base["final_edge"]},
                            {"rho": c["final_rho"], "edge": c["final_edge"]},
                        ),
                    }
                )
    closure = scaling_rows(
        [c for c in cases if "closure" in tasks[c["source"].removesuffix(".npz")]["roles"]], population=True
    )
    return clocks, comparisons, snapshots, numerics, closure


def micro_tables(records, protocol):
    conditional, pooled, contrasts, branches, snapshots = [], [], [], [], []
    grouped = defaultdict(list)
    hashes = defaultdict(set)
    process_controls = defaultdict(list)
    for record in records:
        task, result = record["task"], record["result"]
        p = task["parameters"]
        hashes[(p["population"], task["initial_seed"])].add(result["initial_hash"])
        grouped[key(p) + (task["initial"],)].append(record)
        process_controls[key(p, ("bins", "quadrature")) + (task["initial"], task["branch"])].append(record)
        micro, pair = result["frames"], result["pair"]["frames"]
        mt = np.array([f["time"] for f in micro]) * p["alpha"]
        pt = np.array([f["time"] for f in pair]) * p["alpha"]
        row = {
            **metadata(p),
            "id": task["id"],
            "initial": task["initial"],
            "branch": task["branch"],
            "horizon_tau": float(mt[-1]),
            "micro_variance": micro[-1]["exact_variance"],
            "micro_discordant": micro[-1]["exact_discordant"],
            "pair_variance": pair[-1]["variance"],
            "pair_discordant": pair[-1]["discordant"],
            "micro_pairing_excess": pairing_excess(micro[-1], p["epsilon"]),
            "pair_pairing_excess": pairing_excess(pair[-1], p["epsilon"]),
            **compare_fields(micro[-1], pair[-1]),
        }
        for n in ("node", "edge"):
            row[n + "_energy_relative_gap"] = curve_distance(
                pt, [f[n + "_energy"] for f in pair], mt, [f["exact_" + n + "_energy"] for f in micro]
            )["relative_max"]
        branches.append(row)
        pair_by_step = {f["step"]: f for f in pair if np.asarray(f.get("edge", [])).size}
        for f in micro:
            if np.asarray(f.get("edge", [])).size and f["step"] in pair_by_step:
                snapshots.append(
                    {
                        **metadata(p),
                        "initial": task["initial"],
                        "branch": task["branch"],
                        "tau": f["time"] * p["alpha"],
                        **compare_fields(f, pair_by_step[f["step"]]),
                    }
                )
    if any(len(h) != 1 for h in hashes.values()):
        raise ValueError("initial state changed under recommendation/mesh/process control")
    for members in process_controls.values():
        ref = members[0]["result"]["frames"]
        for record in members[1:]:
            for n in ("node", "edge"):
                a = np.array([(f["time"], f["exact_" + n + "_energy"]) for f in ref])
                b = np.array([(f["time"], f["exact_" + n + "_energy"]) for f in record["result"]["frames"]])
                if (
                    a.shape != b.shape
                    or not np.array_equal(a[:, 0], b[:, 0])
                    or not np.allclose(a[:, 1], b[:, 1], atol=1e-12, rtol=0)
                ):
                    raise ValueError("mesh/quadrature changed the underlying microscopic process")
    cohort_samples = {}
    population_groups = defaultdict(list)
    for group, members in sorted(grouped.items()):
        members.sort(key=lambda r: r["task"]["branch"])
        task = members[0]["task"]
        p = task["parameters"]
        first = members[0]["result"]
        time = np.array([f["time"] for f in first["frames"]]) * p["alpha"]
        matrices = {
            n: np.array([[f["exact_" + n + "_energy"] for f in r["result"]["frames"]] for r in members])
            for n in ("node", "edge")
        }
        if len({r["task"]["process_seed"] for r in members}) != len(members):
            raise ValueError("duplicate process seeds")
        if any(
            not np.array_equal(time, np.array([f["time"] for f in r["result"]["frames"]]) * p["alpha"]) for r in members
        ):
            raise ValueError("cohort time grids differ")
        pair = first["pair"]["frames"]
        pt = np.array([f["time"] for f in pair]) * p["alpha"]
        reference = {n: np.interp(time, pt, [f[n + "_energy"] for f in pair]) for n in ("node", "edge")}

        def pair_hash(result):
            return (
                result["pair_frames_fingerprint"]
                if "pair_frames_fingerprint" in result
                else content_fingerprint(result["pair"]["frames"])
            )

        if any(pair_hash(r["result"]) != pair_hash(first) for r in members):
            raise ValueError("deterministic pair trajectory changed with process seed")
        weights = bootstrap_weights(len(members), protocol)
        md = {
            **metadata(p),
            "initial": task["initial"],
            "initial_hash": first["initial_hash"],
            "processes": len(members),
            "process_seeds": ",".join(str(r["task"]["process_seed"]) for r in members),
            "uncertainty": "conditional on fixed initial graph/opinions",
        }
        pair_clocks = {n: release_clock(time, y) for n, y in reference.items()}
        for plateaus in (None, {n: c.plateau for n, c in pair_clocks.items()}):
            stat, sample = clock_statistics(
                time,
                matrices["node"],
                matrices["edge"],
                weights,
                confidence=protocol["confidence_level"],
                plateaus=plateaus,
            )
            row = {
                **md,
                **stat,
                "time_units": "tau=alpha*t",
                "micro_plateau_confirmed": all(check(time, m.mean(0), protocol) for m in matrices.values()),
                "pair_plateau_confirmed": all(check(time, y, protocol) for y in reference.values()),
            }
            for n, c in pair_clocks.items():
                row["pair_" + n + "_mean"] = c.area
                row[n + "_relative_error"] = abs(stat[n + "_mean"] / c.area - 1) if c.area > 0 else np.nan
            row["pair_edge_to_node"] = (
                pair_clocks["edge"].area / pair_clocks["node"].area if pair_clocks["node"].area > 0 else np.nan
            )
            conditional.append(row)
            if plateaus is None:
                cohort_samples[group] = (row, sample)
        population_groups[group[:-1]].append((md, time, matrices, members))
    # Same resampling draws across recommendations preserve paired contrasts.
    for group, (row, sample) in cohort_samples.items():
        if row["recommender"] == "random":
            continue
        reference = list(group)
        reference[1] = "random"
        reference[2] = 1.0
        other = cohort_samples.get(tuple(reference))
        if other:
            baseline, base_sample = other
            if row["process_seeds"] != baseline["process_seeds"]:
                continue  # Partial designs cannot claim paired recommendation intervals.
            for metric in ("node", "edge", "edge_to_node"):
                contrasts.append(
                    {
                        **{k: row[k] for k in KEYS},
                        "initial": row["initial"],
                        "metric": metric,
                        "difference": row[metric + "_mean"] - baseline[metric + "_mean"],
                        "reference": "Random; shared initial state and paired process seeds",
                        **paired_difference(
                            sample[metric],
                            base_sample[metric],
                            confidence=protocol["confidence_level"],
                            identifiable=row["processes"] > 1,
                        ),
                    }
                )
    branch_by_id = {r["id"]: r for r in branches}
    for members in population_groups.values():
        md, time, _, _ = members[0]
        if any(not np.array_equal(time, m[1]) for m in members):
            raise ValueError("population initial-state time grids differ")
        sizes = [len(m[3]) for m in members]
        weights = hierarchical_weights(sizes, protocol)
        matrices = {n: np.concatenate([m[2][n] for m in members]) for n in ("node", "edge")}
        # Point estimate gives equal weight to initial states, including partial designs.
        means = {n: np.array([m[2][n].mean(0) for m in members]) for n in ("node", "edge")}
        stat, _ = clock_statistics(
            time, matrices["node"], matrices["edge"], weights, confidence=protocol["confidence_level"]
        )
        points, _ = clock_statistics(time, means["node"], means["edge"], np.full((1, len(members)), 1 / len(members)))
        row = {
            **{k: md[k] for k in KEYS},
            **stat,
            "initials": len(members),
            "branches": sum(sizes),
            "uncertainty": "independent initial states, then conditional processes",
            "plateau_confirmed": all(check(time, v.mean(0), protocol) for v in means.values()),
        }
        for metric in ("node", "edge", "edge_to_node"):
            row[metric + "_mean"] = points[metric + "_mean"]
            if len(members) < 2:
                row[metric + "_lower"] = row[metric + "_upper"] = None
        for metric in (
            "micro_variance",
            "micro_discordant",
            "pair_variance",
            "pair_discordant",
            "micro_pairing_excess",
            "pair_pairing_excess",
            "node_w1",
            "edge_hellinger",
        ):
            clusters = [[branch_by_id[r["task"]["id"]][metric] for r in m[3]] for m in members]
            stats = clustered_interval(clusters, weights, confidence=protocol["confidence_level"])
            row.update({metric + "_" + k: v for k, v in stats.items()})
        pooled.append(row)
    return conditional, pooled, contrasts, branches, snapshots


def analyze(directory):
    """Validate coverage/provenance and write standalone tables and an evidence report."""
    plan = json.loads((directory / "plan.json").read_text())
    if fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan["plan_fingerprint"]:
        raise ValueError("evidence plan hash mismatch")
    tasks = {t["id"]: t for t in plan["tasks"]}
    protocol = plan["analysis_protocol"]
    cases = list(cached_cases(directory / "measure"))
    records = []
    coverage = []
    provenance = []
    pair_provenance = {}
    for task in tasks.values():
        suffix = ".npz" if task["kind"] == "measure" else ".msgpack.lz4"
        raw = directory / task["kind"] / "raw"
        path = raw / (task["id"] + suffix) if task["kind"] == "measure" else record_path(raw, task["id"])
        coverage.append(
            {"id": task["id"], "kind": task["kind"], "roles": ",".join(task["roles"]), "complete": path.is_file()}
        )
        if not path.is_file():
            continue
        if task["kind"] == "micro":
            record = read_micro(path)
            if record["task"] != task:
                raise ValueError("raw micro task/plan mismatch")
            reference = record["result"].get("pair_ref")
            if reference:
                pair_provenance[reference["key"]] = {
                    "id": reference["key"],
                    "sha256": reference["sha256"],
                    "path": f"micro/pairs/{reference['key']}.msgpack.lz4",
                }
            records.append(compact_record(record))
        else:
            with np.load(path, allow_pickle=False) as archive:
                if json.loads(str(archive["metadata"])) != task:
                    raise ValueError("raw measure task/plan mismatch")
        provenance.append({"id": task["id"], "sha256": sha(path), "path": str(path.relative_to(directory))})
    if not cases or not records:
        raise FileNotFoundError("evidence requires newly generated measure and micro results")
    clocks, collapse, states, numerics, closure = measure_tables(cases, tasks, protocol)
    conditional, population, contrasts, branches, micro_states = micro_tables(records, protocol)
    for name, rows in (
        ("coverage", coverage),
        ("provenance", provenance),
        ("pair_provenance", list(pair_provenance.values())),
        ("mean_release", clocks),
        ("scaling", collapse),
        ("scaling_states", states),
        ("numerics", numerics),
        ("nominal_population_closure", closure),
        ("conditional_means", conditional),
        ("population", population),
        ("recommendation_contrasts", contrasts),
        ("micro_branches", branches),
        ("micro_states", micro_states),
    ):
        path = directory / "tables" / (name + ".csv")
        if rows:
            write_csv(path, rows)
        else:
            path.unlink(missing_ok=True)  # No stale rows in a partial/legacy analysis.
    summary = {
        "schema": 1,
        "profile": plan["profile"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "planned": len(tasks),
        "completed": len(provenance),
        "missing": [r["id"] for r in coverage if not r["complete"]],
        "measure": len(cases),
        "early_stopped_measure": sum(bool(c["diagnostics"].get("converged", False)) for c in cases),
        "observed_measure_horizon_range": [
            float(min(c["tau"][-1] for c in cases)),
            float(max(c["tau"][-1] for c in cases)),
        ],
        "micro": len(records),
        "conditional_cohorts": len(conditional) // 2,
        "confirmed_measure_rows": sum(r["plateau_confirmed"] for r in clocks),
        "analysis_protocol": protocol,
        "bindings": {k: json.loads((directory / k / "binding.json").read_text()) for k in ("measure", "micro")},
        "initial_and_process_controls": "passed",
        "claim_scope": "observed restricted means and finite-N robustness; collapse is approximate small-update evidence",
    }
    atomic_json(directory / "report.json", summary)
    text = f"""# Fresh evidence report\n\nProfile: {plan["profile"]}. Completed {len(provenance)}/{len(tasks)} tasks; missing {len(summary["missing"])}.\n\nThe clock is the area under normalized remaining energy in tau=alpha*t. T50 is retained as a diagnostic. Prefix areas use the same long-window endpoint reference. Endpoint estimates and their two-window stability checks are reported separately; unconfirmed curves do not establish infinite mean release times.\n\nConditional intervals resample process branches at a fixed initial state. Population intervals resample independent initial states and then their branches. Recommendation contrasts share initial states and process seed labels. Own endpoint and fixed pair endpoint references appear separately.\n\nScaling compares raw energies at fixed r=q/alpha and numerical resolution, plus selected rho/edge states. Grid, quadrature and dt checks are separate. These support a leading small-update law; they do not assert exact finite-step equivalence. Finite-N comparisons retain k=15, R=10 and the uniform initial law. Variance and discordant-edge fraction describe polarization and segregation; edge distributions do not measure graph modularity.\n\nAll raw inputs are listed in tables/provenance.csv with hashes. Coverage is in tables/coverage.csv. No historical run IDs, datasets or manuscript artifacts are required. CAS 2026 remains the external empirical motivation, not a dataset regenerated by this design.\n\nThe smoke profile checks software paths only. Formal evidence requires the paper profile and assessment of plateau, tail, discretization and finite-N diagnostics.\n\nFigures: mean_release, scaling_collapse, matched_micro_means, population_robustness, numerical_controls. Numerical controls and scaling state errors are candidates for supplementary material.\n"""
    (directory / "report.md").write_text(text)
    print({k: summary[k] for k in ("planned", "completed", "measure", "micro", "conditional_cohorts")}, flush=True)
    return summary
