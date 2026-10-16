"""Offline assembly from this design's frozen records, without historical inputs."""

import csv
import inspect
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from common.experiments.artifacts import ArtifactCache
from common.experiments.lifecycle import session_lock
from common.experiments.plans import coverage as task_coverage
from common.io.json import atomic_json, fingerprint, sha
from common.io.records import content_fingerprint
from common.statistics import bootstrap_weights, clustered_interval, hierarchical_weights, paired_difference
from theory.release import release_clock
from theory.release.ensemble import clock_statistics, plateau_check
from theory.scaling import curve_distance
from theory.types import MicroFrame, PairFrame, ResolvedMicroRecord
from theory.validation import compare_fields

from ..data import write_csv
from ..scaling.analyze import scaling_rows
from ..types import EvidenceReport
from .bundle import Bundle, csv_rows, csv_stream, export_raw
from .observations import Tables, matched_bands, population_variation, representative_fields

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


def retain_projection(frame: MicroFrame | PairFrame, endpoint: bool) -> bool:
    """Keep state fields at declared snapshots or the observed endpoint."""
    if np.asarray(frame.get("edge", [])).size or endpoint:
        if "rho" in frame:
            frame["rho"] = np.asarray(frame["rho"], dtype=float)
        if "edge" in frame:
            frame["edge"] = np.asarray(frame["edge"], dtype=float)
        return True
    return False


def compact_record(record: ResolvedMicroRecord) -> ResolvedMicroRecord:
    """Keep full fields only at snapshots/endpoints, not at every energy sample."""
    result = record["result"]
    if "pair_frames_fingerprint" not in result:
        result["pair_frames_fingerprint"] = content_fingerprint(result["pair"]["frames"])
    micro = result["frames"]
    for i, frame in enumerate(micro):
        if not retain_projection(frame, i == len(micro) - 1):
            micro[i] = {
                "time": frame["time"],
                "step": frame["step"],
                "exact_node_energy": frame["exact_node_energy"],
                "exact_edge_energy": frame["exact_edge_energy"],
            }
    pair = result["pair"]["frames"]
    for i, prediction in enumerate(pair):
        if not retain_projection(prediction, i == len(pair) - 1):
            pair[i] = {
                "time": prediction["time"],
                "step": prediction["step"],
                "node_energy": prediction["node_energy"],
                "edge_energy": prediction["edge_energy"],
            }
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
        baseline_key = list(group)
        baseline_key[1] = "random"
        baseline_key[2] = 1.0
        other = cohort_samples.get(tuple(baseline_key))
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


def recommendation_rows(clocks):
    endpoint = [r for r in clocks if "effects" in r["roles"].split(",") and r["is_observed_endpoint"]]
    groups: dict[tuple, dict[str, Any]] = defaultdict(dict)
    for r in endpoint:
        groups[tuple((k, r[k]) for k in KEYS)][r["channel"]] = r
    ratios = {}
    for identity, channels in groups.items():
        if {"node", "edge"} <= channels.keys():
            node, edge = channels["node"], channels["edge"]
            value = (
                np.log(edge["mean_tau"] / node["mean_tau"]) if min(edge["mean_tau"], node["mean_tau"]) > 0 else np.nan
            )
            ratios[identity] = {
                **dict(identity),
                "delta": float(value),
                "confirmed": bool(node["plateau_confirmed"] and edge["plateau_confirmed"]),
                "observed_horizon_tau": node["reference_horizon_tau"],
            }
    for identity, row in ratios.items():
        base = dict(identity)
        base.update(recommender="random", steepness=1.0)
        reference = ratios.get(tuple((k, base[k]) for k in KEYS))
        row["shift"] = row["delta"] - reference["delta"] if reference else np.nan
        row["shift_confirmed"] = bool(row["confirmed"] and reference and reference["confirmed"])
    return list(ratios.values())


def merge_tables(parts, destination):
    files = defaultdict(list)
    for part in parts:
        for path in part.glob("*.csv*"):
            files[path.name].append(path)
    for name, paths in files.items():
        columns: list[str] = []
        for path in paths:
            first = next(csv_rows(path), None)
            if first:
                columns.extend(k for k in first if k not in columns)
        if not columns:
            continue
        with csv_stream(destination / name, "wt") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            for path in paths:
                writer.writerows(csv_rows(path))


def code_identity(functions):
    from common.numerics.distributions import hellinger
    from theory import variation

    from .bundle import number

    libraries = (
        bootstrap_weights,
        variation.with_loio,
        clock_statistics,
        release_clock,
        curve_distance,
        compare_fields,
        hellinger,
    )

    return {
        "functions": {f.__name__: fingerprint(inspect.getsource(f)) for f in functions},
        "readers": {
            f.__qualname__: fingerprint(inspect.getsource(f))
            for f in (
                Bundle.case,
                Bundle.record,
                Bundle.frames,
                Bundle.matrices,
                number,
                compact_record,
                retain_projection,
            )
        },
        "libraries": {f.__module__: sha(Path(inspect.getfile(f))) for f in libraries},
        "parameter_columns": KEYS,
        "numpy_version": np.__version__,
    }


def scaling_display(rows):
    grouped: dict[tuple, dict[str, Any]] = {}
    keys = ("dynamics", "recommender", "steepness", "epsilon", "r", "alpha", "reference_alpha")
    for row in rows:
        identity = tuple(row[k] for k in keys)
        point = grouped.setdefault(identity, {**{k: row[k] for k in keys}, "relative_max": 0.0})
        point["relative_max"] = max(point["relative_max"], row["relative_max"])
    return list(grouped.values())


def analyze(directory, output=None, *, force=False):
    """Incremental analysis from raw records or portable CSV; force recomputes all blocks."""
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory
    with session_lock(output / ".analysis-control"):
        return compute_analysis(directory, output, force=force)


def compute_analysis(directory, output, *, force):
    if any((directory / kind / "raw").is_dir() for kind in ("measure", "micro")):
        export_raw(directory, output, force=force)
    elif (directory / "data_manifest.json").is_file():
        if output != directory:
            for name in ("plan.json", "data_manifest.json"):
                shutil.copy2(directory / name, output / name)
            shutil.copytree(directory / "data", output / "data", dirs_exist_ok=True)
            if (directory / "analysis_cache").is_dir():
                shutil.copytree(directory / "analysis_cache", output / "analysis_cache", dirs_exist_ok=True)
    else:
        raise FileNotFoundError("analyze needs native records or a portable CSV observations bundle")
    bundle = Bundle(output)
    plan, tasks = bundle.plan, bundle.tasks
    protocol = plan["analysis_protocol"]
    cache = ArtifactCache(output / "analysis_cache", force=force)
    stage = Path(tempfile.mkdtemp(prefix=".tables-", dir=output))
    parts = []
    horizons = []
    early = 0
    try:
        measure_groups = defaultdict(list)
        micro_groups = defaultdict(list)
        for t in tasks.values():
            if t["id"] not in bundle.manifest["records"]:
                continue
            p = t["parameters"]
            if t["kind"] == "measure":
                measure_groups[
                    tuple(p[k] for k in ("dynamics", "recommender", "steepness", "epsilon", "ratio"))
                ].append(t)
            else:
                micro_groups[tuple(p[k] for k in KEYS if k not in ("recommender", "steepness"))].append(t)
        for members in measure_groups.values():

            def produce(path):
                cases = []
                for t in members:
                    fields: set[str] = set()
                    if "scaling" in t["roles"]:
                        fields.update(("snapshot_time", "snapshot_rho", "snapshot_edge"))
                    if "numerics" in t["roles"]:
                        fields.update(("final_rho", "final_edge"))
                    cases.append(bundle.case(t, fields=fields))
                rows = measure_tables(cases, tasks, protocol)
                for name, values in zip(
                    ("mean_release", "scaling", "scaling_states", "numerics", "nominal_population_closure"), rows
                ):
                    if values:
                        write_csv(path / (name + ".csv"), values)
                atomic_json(
                    path / "summary.json",
                    {
                        "horizons": [float(c["tau"][-1]) for c in cases],
                        "early": sum(bool(c["diagnostics"].get("converged", False)) for c in cases),
                    },
                )

            part = cache.materialize(
                "measure",
                {
                    "tasks": members,
                    "files": bundle.dependencies(members),
                    "protocol": protocol,
                    "code": code_identity((measure_tables, clock_rows, check, scaling_rows)),
                },
                produce,
            )
            parts.append(part)
            info = json.loads((part / "summary.json").read_text())
            horizons.extend(info["horizons"])
            early += info["early"]
        for index, members in enumerate(micro_groups.values()):
            loaded = {}

            def get_records(fields=True):
                if fields not in loaded:
                    pair_cache: dict[str, Any] = {}
                    loaded[fields] = [bundle.record(t, pair_cache=pair_cache, fields=fields) for t in members]
                    if fields:
                        loaded[fields] = [compact_record(r) for r in loaded[fields]]
                return loaded[fields]

            for name, function, fields, settings in (
                ("micro_summaries", micro_tables, True, protocol),
                (
                    "matched_curves",
                    matched_bands,
                    False,
                    {k: protocol[k] for k in ("bootstrap_seed", "bootstrap_samples", "confidence_level")},
                ),
                (
                    "representative_fields",
                    representative_fields,
                    True,
                    {k: protocol[k] for k in ("main_snapshot_tau", "matched_population")},
                ),
                ("population_variation", population_variation, True, {}),
            ):
                enabled = (
                    name == "micro_summaries"
                    or (name == "matched_curves" and any("matched" in t["roles"] for t in members))
                    or (name == "population_variation" and any("population" in t["roles"] for t in members))
                    or (
                        name == "representative_fields"
                        and any(
                            "population" in t["roles"]
                            and t["initial"] == 0
                            and t["parameters"]["population"] == protocol["matched_population"]
                            and t["parameters"]["dynamics"] == "hk"
                            and t["parameters"]["recommender"] == "opinion_random"
                            and t["parameters"]["steepness"] == 4
                            for t in members
                        )
                    )
                )
                if not enabled:
                    continue

                def produce(path):
                    records = get_records(fields)
                    if name == "micro_summaries":
                        for table, values in zip(
                            (
                                "conditional_means",
                                "population",
                                "recommendation_contrasts",
                                "micro_branches",
                                "micro_states",
                            ),
                            micro_tables(records, protocol),
                        ):
                            if values:
                                write_csv(path / (table + ".csv"), values)
                    else:
                        with Tables(path) as tables:
                            if name == "population_variation":
                                cast(Callable[..., Any], function)(records, tables)
                            else:
                                cast(Callable[..., Any], function)(records, protocol, tables)

                functions: tuple[Callable[..., Any], ...] = (function,)
                if name == "micro_summaries":
                    functions += (key, metadata, pairing_excess, check)
                part = cache.materialize(
                    name,
                    {
                        "tasks": members,
                        "files": bundle.dependencies(members, fields=fields),
                        "protocol": settings,
                        "code": code_identity(functions),
                    },
                    produce,
                )
                parts.append(part)
            print(
                f"CSV analysis {index + 1}/{len(micro_groups)} configurations; cache {cache.hits} reused, {cache.computed} computed",
                flush=True,
            )
        merge_tables(parts, stage)
        from common.io.tables import numeric_rows

        clocks = numeric_rows(stage / "mean_release.csv") if (stage / "mean_release.csv").is_file() else []
        # CSV booleans retain their explicit strings; restore them before derivation.
        for r in clocks:
            for k in ("is_observed_endpoint", "plateau_confirmed"):
                r[k] = r[k] == "True"
        for name, values in (
            ("recommendation", recommendation_rows(clocks)),
            (
                "scaling_display",
                scaling_display(numeric_rows(stage / "scaling.csv")) if (stage / "scaling.csv").exists() else [],
            ),
        ):
            if values:
                write_csv(stage / (name + ".csv"), values)
        provenance = [r for r in bundle.manifest["provenance"] if r["kind"] != "pair"]
        pairs = [r for r in bundle.manifest["provenance"] if r["kind"] == "pair"]
        coverage = [
            {
                "id": t["id"],
                "kind": t["kind"],
                "roles": ",".join(t["roles"]),
                "complete": t["id"] in bundle.manifest["records"],
            }
            for t in tasks.values()
        ]
        for name, rows in (("coverage", coverage), ("provenance", provenance), ("pair_provenance", pairs)):
            if rows:
                write_csv(stage / (name + ".csv"), rows)
        conditional_count = (
            sum(1 for _ in csv_rows(stage / "conditional_means.csv"))
            if (stage / "conditional_means.csv").is_file()
            else 0
        )
        summary: EvidenceReport = {
            "schema": 1,
            "profile": plan["profile"],
            "plan_fingerprint": plan["plan_fingerprint"],
            **task_coverage(plan["tasks"], bundle.manifest["records"]),
            "measure": sum(t["kind"] == "measure" and t["id"] in bundle.manifest["records"] for t in tasks.values()),
            "micro": sum(t["kind"] == "micro" and t["id"] in bundle.manifest["records"] for t in tasks.values()),
            "early_stopped_measure": early,
            "observed_measure_horizon_range": [min(horizons), max(horizons)] if horizons else [],
            "conditional_cohorts": conditional_count // 2,
            "confirmed_measure_rows": sum(r["plateau_confirmed"] for r in clocks),
            "analysis_protocol": protocol,
            "bindings": bundle.manifest["bindings"],
            "initial_and_process_controls": "passed",
            "claim_scope": "finite-window response, conditional validation, finite-N structure and variability",
        }
        old = output / ".previous-tables"
        if old.exists():
            shutil.rmtree(old)
        if (output / "tables").exists():
            os.replace(output / "tables", old)
        os.replace(stage, output / "tables")
        if old.exists():
            shutil.rmtree(old)
        atomic_json(output / "report.json", summary)
        (output / "report.md").write_text(
            f"# Theory analysis\n\nProfile: {plan['profile']}; completed {summary['completed']}/{summary['planned']}.\n\n"
            "data/ preserves all recorded scalar trajectories, actual time grids and native rho/edge fields as CSV. "
            "tasks.csv retains initial/process identities and shared-pair references; provenance paths are never read by plot.\n\n"
            "tables/ contains conditional bootstrap curves, finite-window own-reference areas, numerical comparisons, "
            "population structure and weighted CDF variation components. W/I/B² intervals are LOIO ranges, not 95% CIs. "
            "Negative corrections and near-consensus branches are retained. Structure intervals resample whole initial groups.\n\n"
            f"Incremental cache: {cache.hits} reused, {cache.computed} computed blocks. "
            "Input hashes, scientific code and analysis protocols determine reuse. --force recomputes all blocks.\n"
        )
        from theory import variation

        from . import bundle as bundle_module
        from . import observations

        sources = [Path(__file__), Path(bundle_module.__file__), Path(observations.__file__), Path(variation.__file__)]
        atomic_json(
            output / "analysis_manifest.json",
            {
                "schema": 1,
                "plan_fingerprint": plan["plan_fingerprint"],
                "data_manifest_sha256": sha(output / "data_manifest.json"),
                "report_sha256": sha(output / "report.json"),
                "tables": {str(f.relative_to(output)): sha(f) for f in sorted((output / "tables").glob("*.csv*"))},
                "sources": {str(p): sha(p) for p in sources},
                "cache": {"reused": cache.hits, "computed": cache.computed, "forced": force},
                "variation_protocol": {
                    "norm": "dx sum(node CDF²); dx² sum(joint edge CDF²)",
                    "components": "MSE=W+B²; I=V_P+H+2C",
                    "uncertainty": "LOIO range; not a confidence interval",
                    "negative_corrections": "retained",
                    "structure_intervals": "95% whole-initial-group bootstrap; exploratory",
                },
                "portable": True,
            },
        )
        print({k: summary[k] for k in ("planned", "completed", "measure", "micro")}, flush=True)
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)
