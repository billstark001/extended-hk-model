"""Offline extraction and study assembly; mathematical estimators live in lifted."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common.experiments.lifecycle import session_lock
from common.experiments.plans import coverage, read_plan
from common.io.json import atomic_json, sha
from common.io.tables import table
from lifted.commitment import ensemble_cdf, timing
from lifted.committor import estimate, terminal_boundary, total_variation
from lifted.diagnostics import factor_summary, increment_factors
from lifted.outcomes import CATEGORIES
from lifted.runtime.records import read, record_path

from .protocol import sources

TABLES = (
    "probabilities",
    "timing",
    "cdf_bounds",
    "runtime_costs",
    "moments",
    "states",
    "mother_paths",
    "noise_controls",
    "measure",
    "closure_error",
    "conditional_validation",
)


def plans(root):
    root = Path(root)
    plan = read_plan(root / "plan.json")
    tasks = {t["id"]: t for t in plan["tasks"]}
    followup = root / "followup/plan.json"
    if followup.exists():
        extra = read_plan(followup)
        if extra["parent_plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("follow-up belongs to a different frozen plan")
        for relative, digest in extra["inputs"].items():
            if sha(root / relative) != digest:
                raise ValueError("follow-up input changed")
        tasks.update({t["id"]: t for t in extra["tasks"]})
    return plan, tasks


def available(root, tasks, *, inputs=None):
    if not (Path(root) / "binding.json").exists():
        if any(record_path(root, t["id"]).exists() for t in tasks.values()):
            raise ValueError("raw records lack their executable binding")
        return
    binding = json.loads((Path(root) / "binding.json").read_text())
    for task in tasks.values():
        if inputs is not None and task["id"] not in inputs:
            continue
        if record_path(root, task["id"]).exists():
            result = read(root, task, binding)
            if inputs is not None and sha(record_path(root, task["id"])) != inputs[task["id"]]:
                raise ValueError("analysis snapshot input changed")
            yield task, result
        elif inputs is not None:
            raise ValueError("analysis snapshot input disappeared")


def status(root):
    root = Path(root)
    plan, tasks = plans(root)
    completed = [task["id"] for task, _ in available(root, tasks)]
    analysis = root / "analysis/manifest.json"
    plots = root / "analysis/plot_manifest.json"
    inputs = {
        str(record_path(root, identifier).relative_to(root)): sha(record_path(root, identifier))
        for identifier in completed
    }
    analyzed = json.loads(analysis.read_text()) if analysis.exists() else {}
    plotted = json.loads(plots.read_text()) if plots.exists() else {}
    analysis_current = (
        bool(analyzed)
        and analyzed["inputs"] == inputs
        and analyzed["plan_fingerprint"] == plan["plan_fingerprint"]
        and analyzed.get("followup_sha256")
        == (sha(root / "followup/plan.json") if (root / "followup/plan.json").exists() else None)
        and all(
            (root / "analysis" / name).is_file() and sha(root / "analysis" / name) == digest
            for name, digest in analyzed["files"].items()
        )
    )
    return {
        "profile": plan["profile"],
        "plan_fingerprint": plan["plan_fingerprint"],
        **coverage(tasks.values(), completed),
        "formal_started": plan["profile"] == "formal" and bool(completed),
        "analysis_current": analysis_current,
        "plots_current": analysis_current
        and bool(plotted)
        and plotted["analysis_manifest_sha256"] == sha(analysis)
        and all(
            (root / "analysis" / name).is_file() and sha(root / "analysis" / name) == digest
            for name, digest in plotted["files"].items()
        ),
    }


def query_task(task, tasks):
    while task["kind"].endswith("extend"):
        task = tasks[task["source_task"]]
    return task


def group_key(task):
    p = task["parameters"]
    return (
        task["kind"],
        task["source_task"],
        task["source_step"],
        p["population"],
        p["bins"],
        p["ratio"],
        task["variant"],
        task["layer"],
        task["cohort"],
    )


def is_query(task):
    return task["kind"] in ("micro_probe", "meso_probe") and task["variant"] == "stochastic"


def probability_groups(root, plan=None, *, tasks=None, inputs=None):
    saved, tasks = plans(root) if tasks is None else (plan, tasks)
    grouped: dict[tuple, Any] = {}
    boundaries: dict[tuple, str] = {}
    for task in tasks.values():
        if is_query(task):
            group = grouped.setdefault(group_key(task), {"task": task, "paths": {}, "planned": 0})
            group["planned"] += len(task["replicates"])
    for task, result in available(root, tasks, inputs=inputs):
        for row in result["states"]:
            if row["category"] in CATEGORIES:
                boundaries[(task["id"], row["step"])] = row["category"]
        original = query_task(task, tasks)
        if not is_query(original):
            continue
        key = group_key(original)
        group = grouped[key]
        for row in result["paths"]:
            index = (original["id"], row["replicate"])
            previous = group["paths"].get(index)
            if previous is None or previous["step"] < row["step"]:
                group["paths"][index] = row
    output = []
    # Reserve the prespecified 2*9*(micro+meso) window queries and six independent
    # validations before seeing outcomes. Partial exports and adaptive selection
    # must not change the core's simultaneous correction.
    base_family = len({group_key(t) for t in saved["tasks"] if is_query(t)})
    reserve = saved["protocol"].get("followup_query_reserve", 42) if saved["profile"] == "formal" else 0
    family = max(1, base_family + reserve)
    if len(grouped) > family:
        raise ValueError("query family exceeds the prespecified simultaneous budget")
    for group in grouped.values():
        task, paths = group["task"], list(group["paths"].values())
        p = task["parameters"]
        ancestor = task
        while ancestor["source_task"]:
            ancestor = tasks[ancestor["source_task"]]
        probability = estimate(
            [r["category"] for r in paths],
            confidence=saved["protocol"]["confidence"],
            family_size=family,
            pending=group["planned"] - len(paths),
        )
        boundary = None
        if task["kind"] == "micro_probe" or tasks[task["source_task"]]["kind"].startswith("meso_"):
            boundary = boundaries.get((task["source_task"], task["source_step"]))
        if paths and all(r["step"] == task["source_step"] and r["stop_reason"] == "geometric_terminal" for r in paths):
            boundary = paths[0]["category"]
        if boundary is not None:
            probability = terminal_boundary(probability, boundary)
        output.append(
            {
                **probability,
                "probability_method": "terminal_boundary" if boundary else "independent_continuations",
                "kind": task["kind"],
                "source_task": task["source_task"],
                "step": task["source_step"],
                "tau": task["source_step"] * p["alpha"] * p["dt"],
                "ratio": p["ratio"],
                "population": p["population"],
                "bins": p["bins"],
                "variant": task["variant"],
                "layer": task["layer"],
                "cohort": task["cohort"],
                "initial": task["initial"],
                "mother": task["mother"],
                "initial_law": ancestor["initial_law"],
                "bridge_agents": ancestor["bridge_agents"],
                "representative_task": task["id"],
                "family_size": family,
            }
        )
    return output


def analyze(root):
    with session_lock(Path(root) / "analysis"):
        return _analyze(root)


def _analyze(root):
    root = Path(root)
    followup_path = root / "followup/plan.json"
    followup_sha = sha(followup_path) if followup_path.exists() else None
    plan, tasks = plans(root)
    if (sha(followup_path) if followup_path.exists() else None) != followup_sha:
        raise ValueError("follow-up plan changed while capturing the analysis snapshot; analyze again")
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    inputs = {t["id"]: sha(record_path(root, t["id"])) for t in tasks.values() if record_path(root, t["id"]).exists()}
    # One immutable record set for every table, even when execution continues concurrently.
    groups = probability_groups(root, plan, tasks=tasks, inputs=inputs)
    probabilities = []
    for group in groups:
        row = {k: v for k, v in group.items() if k not in ("counts", "estimate", "lower", "upper")}
        for i in range(4):
            for key in ("estimate", "lower", "upper", "counts"):
                row[f"{key}_{i + 1}"] = group[key][i]
        probabilities.append(row)
    table(output / "probabilities.csv", probabilities)
    cohorts = defaultdict(list)
    for group in groups:
        if group["kind"] == "micro_probe" and group["cohort"] in ("population", "smoke", "pilot"):
            cohorts[(group["ratio"], group["population"], group["initial"], group["mother"])].append(group)
    times = []
    for (ratio, n, initial, mother), rows in cohorts.items():
        # Mother zero's t=0 probabilities are shared, not independently resampled.
        if mother and not any(r["tau"] == 0 for r in rows):
            start = [
                g
                for g in groups
                if g["kind"] == "micro_probe"
                and g["ratio"] == ratio
                and g["population"] == n
                and g["initial"] == initial
                and g["mother"] == 0
                and g["tau"] == 0
                and g["cohort"] == rows[0]["cohort"]
            ]
            rows = [*start, *rows]
        rows = sorted(rows, key=lambda r: r["tau"])
        for delta in plan["protocol"]["deltas"]:
            times.append(
                {
                    "ratio": ratio,
                    "population": n,
                    "initial": initial,
                    "mother": mother,
                    **timing([r["tau"] for r in rows], rows, delta=delta),
                }
            )
    table(output / "timing.csv", times)
    cdfs: list[dict] = []
    for ratio, n, delta in sorted({(r["ratio"], r["population"], r["delta"]) for r in times}):
        selected = [r for r in times if (r["ratio"], r["population"], r["delta"]) == (ratio, n, delta)]
        cdfs.extend(
            {"ratio": ratio, "population": n, "delta": delta, **r}
            for r in ensemble_cdf(selected, plan["protocol"]["observation_taus"])
        )
    table(output / "cdf_bounds.csv", cdfs)
    costs, moments, states, mother_paths, controls = [], [], [], [], []
    truth: dict[str, dict] = {}
    traces: list[dict] = []
    equivalence: dict[tuple, dict] = {}
    for task, result in available(root, tasks, inputs=inputs):
        p = task["parameters"]
        if task["kind"] == "meso_probe" and task["variant"] != "stochastic":
            for row in result["paths"]:
                final = row.get("final", {})
                controls.append(
                    dict(
                        task=task["id"],
                        ratio=p["ratio"],
                        bins=p["bins"],
                        variant=task["variant"],
                        replicate=row["replicate"],
                        query_tau=task["source_step"] * p["alpha"],
                        observed_tau=(row["step"] - task["source_step"]) * p["alpha"],
                        planned_window_tau=task["horizon_steps"] * p["alpha"],
                        category=row["category"],
                        **{k: final[k] for k in ("node_energy", "edge_energy", "variance", "discordant") if k in final},
                    )
                )
        original = query_task(task, tasks)
        if original["kind"] == "micro_tail":
            row = result["paths"][0]
            previous = truth.get(original["source_task"])
            if previous is None or previous["step"] < row["step"]:
                truth[original["source_task"]] = row
        if task["kind"] in ("micro_atlas", "meso_atlas"):
            for row in result["paths"]:
                mother_paths.append(
                    dict(
                        task=task["id"],
                        kind=task["kind"],
                        initial=task["initial"],
                        mother=task["mother"],
                        ratio=p["ratio"],
                        population=p["population"],
                        cohort=task["cohort"],
                        absorption_tau=row["first_hit_step"] * p["alpha"] if "first_hit_step" in row else float("inf"),
                        observed_through_tau=row["step"] * p["alpha"],
                    )
                )
        costs.append(
            dict(
                task=task["id"],
                kind=task["kind"],
                ratio=p["ratio"],
                population=p["population"],
                bins=p["bins"],
                variant=task["variant"],
                layer=task["layer"],
                paths=len(task["replicates"]),
                horizon_steps=task["horizon_steps"],
                source_step=task["source_step"],
                executed_steps=result["executed_steps"],
                elapsed_seconds=result["elapsed_seconds"],
                wall_seconds=result["wall_seconds"],
                unresolved=sum(r["category"] == "unresolved" for r in result["paths"]),
            )
        )
        for row in result["states"]:
            states.append(
                {
                    "task": task["id"],
                    "kind": task["kind"],
                    "initial": task["initial"],
                    "mother": task["mother"],
                    "ratio": p["ratio"],
                    "tau": row["step"] * p["alpha"],
                    **row,
                }
            )
        if task["kind"].endswith("moments"):
            increments = np.array([r["delta"] for r in result["paths"]], dtype=np.float64)
            mean, factor = increment_factors(increments, bins=p["bins"], degree=p["degree"])
            factor_path = output / "moments" / (task["id"] + ".npz")
            factor_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(factor_path, mean=mean, factor=factor)
            moments.append(
                dict(
                    task=task["id"],
                    kind=task["kind"],
                    ratio=p["ratio"],
                    bins=p["bins"],
                    layer=task["layer"],
                    **factor_summary(mean, factor, p["bins"]),
                )
            )
            if task["kind"] == "meso_moments":
                pair_key = (task["source_task"], task["source_step"], p["bins"], tuple(task["replicates"]))
                if task["layer"] == "base":
                    equivalence.setdefault(pair_key, {})["base"] = increments
                else:
                    equivalence.setdefault(pair_key, {})["naive"] = increments
        if result.get("trace"):
            traces.extend(
                dict(
                    task=task["id"],
                    ratio=p["ratio"],
                    **{
                        k: v
                        for k, v in frame.items()
                        if k in ("step", "time", "node_energy", "edge_energy", "variance", "discordant")
                    },
                )
                for frame in result["trace"]["frames"]
            )
    equal = [np.array_equal(v["base"], v["naive"]) for v in equivalence.values() if set(v) == {"base", "naive"}]
    if equal and not all(equal):
        raise ValueError("Random naive/base physical increments differ under identical seeds")
    table(output / "runtime_costs.csv", costs)
    table(output / "moments.csv", moments)
    table(output / "states.csv", states)
    table(output / "mother_paths.csv", mother_paths)
    table(output / "noise_controls.csv", controls)
    table(output / "measure.csv", traces)
    comparisons = []
    for coarse in groups:
        if coarse["kind"] != "meso_probe" or coarse["unresolved"] or coarse["pending_paths"]:
            continue
        matched = [
            m
            for m in groups
            if m["kind"] == "micro_probe"
            and not m["unresolved"]
            and not m["pending_paths"]
            and (m["source_task"], m["step"], m["ratio"], m["cohort"])
            == (coarse["source_task"], coarse["step"], coarse["ratio"], coarse["cohort"])
        ]
        if matched:
            comparisons.append(
                dict(
                    source_task=coarse["source_task"],
                    step=coarse["step"],
                    ratio=coarse["ratio"],
                    bins=coarse["bins"],
                    variant=coarse["variant"],
                    tv=total_variation(matched[0]["estimate"], coarse["estimate"]),
                )
            )
    table(output / "closure_error.csv", comparisons)
    calibration = []
    for group in groups:
        actual = truth.get(group["source_task"])
        if actual is None or actual["category"] == "unresolved" or group["unresolved"] or group["pending_paths"]:
            continue
        target = np.array([float(actual["category"] == c) for c in CATEGORIES])
        predicted = np.asarray(group["estimate"])
        calibration.append(
            dict(
                source_task=group["source_task"],
                step=group["step"],
                kind=group["kind"],
                ratio=group["ratio"],
                bins=group["bins"],
                variant=group["variant"],
                initial=group["initial"],
                mother=group["mother"],
                truth=actual["category"],
                predicted=CATEGORIES[int(np.argmax(predicted))],
                confidence=float(predicted.max()),
                correct=int(CATEGORIES[int(np.argmax(predicted))] == actual["category"]),
                brier=float(np.sum((predicted - target) ** 2)),
            )
        )
    table(output / "conditional_validation.csv", calibration)
    completion = {
        "profile": plan["profile"],
        "plan_fingerprint": plan["plan_fingerprint"],
        **coverage(tasks.values(), inputs),
        "formal_started": plan["profile"] == "formal" and bool(inputs),
    }
    report = {
        **completion,
        "groups": len(groups),
        "probability_coverage": {
            kind: {
                "planned_paths": sum(g["paths"] for g in groups if g["kind"] == kind),
                "pending_paths": sum(g["pending_paths"] for g in groups if g["kind"] == kind),
                "physical_unresolved_paths": sum(g["unresolved"] for g in groups if g["kind"] == kind),
                "exact_boundary_groups": sum(
                    g["probability_method"] == "terminal_boundary" for g in groups if g["kind"] == kind
                ),
            }
            for kind in ("micro_probe", "meso_probe")
        },
        "naive_base_equal_pairs": sum(equal),
        "science_evidence": plan["profile"] == "formal" and not completion["missing"],
        "sampling_unit": "independent initial state, then mother path",
        "timing_scope": "observation grid; simultaneous sampling and unresolved-tail bounds",
        "pilot_timings_only": plan["profile"] == "pilot",
    }
    atomic_json(output / "report.json", report)
    if any(sha(record_path(root, identifier)) != digest for identifier, digest in inputs.items()):
        raise ValueError("analysis snapshot input changed before publication")
    artifacts = [output / (name + ".csv") for name in TABLES]
    artifacts.extend(output / "moments" / (r["task"] + ".npz") for r in moments)
    artifacts.append(output / "report.json")
    atomic_json(
        output / "manifest.json",
        {
            "plan_fingerprint": plan["plan_fingerprint"],
            "followup_sha256": followup_sha,
            "sources": sources(),
            "inputs": {
                str(record_path(root, identifier).relative_to(root)): digest for identifier, digest in inputs.items()
            },
            "files": {str(p.relative_to(output)): sha(p) for p in artifacts},
        },
    )
    return report
