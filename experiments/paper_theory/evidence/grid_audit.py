"""Fixed-ensemble grid audit, with each mesh projected from one physical start.

Run and analyze are separate operations.  Existing raw records are authenticated
and reused; a changed design or runtime requires a new output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.io.json import atomic_json, fingerprint, sha
from common.io.records import read_record, write_record
from common.io.tables import table
from common.numerics.distributions import rebin_matrix
from theory.parameters import Parameters
from theory.runtime.population import binary_result, population_request


def design(epsilons=(0.2, 0.4, 0.8), *, initials=4, processes=16, horizon=30.0):
    """Keep the rapid-rewiring question, using the main design's physical starts."""
    tasks = []
    for epsilon in epsilons:
        for initial in range(initials):
            for kind, bins, branches in (
                ("micro_only", 161, range(processes)),
                *(("pair_only", bins, (0,)) for bins in (81, 161, 241)),
            ):
                for branch in branches:
                    p = Parameters(
                        dynamics="hk", recommender="random", epsilon=epsilon,
                        alpha=0.02, ratio=10.0, degree=15, recommendations=10,
                        bins=bins, population=500, dt=1.0, horizon=horizon,
                        record_tau=0.1, quadrature=33,
                    )
                    task = dict(
                        kind=kind, parameters=asdict(p), initial=initial, branch=branch,
                        initial_seed=17000 + initial,
                        process_seed=24000 + 100 * initial + branch,
                        selection="weighted", request_options={"snapshot_taus": [0.0, horizon]},
                    )
                    task["id"] = kind + "_" + fingerprint(task)[:20]
                    tasks.append(task)
    return {
        "schema": 1, "purpose": "fixed-ensemble rapid-rewiring grid audit",
        "epsilons": list(epsilons), "initials": initials, "processes_per_initial": processes,
        "horizon_tau": horizon, "comparison_bins": 81,
        "initialization": "main-design split opinion/graph RNG; identical physical initial hash across meshes",
        "tasks": tasks,
    }


def freeze(directory, candidate):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != candidate:
            raise ValueError("grid audit design changed; use a new directory")
    else:
        atomic_json(path, candidate)


def execute(directory, plan, workers, backend):
    from ..runtime.build import build

    model = Path(__file__).resolve().parents[3]
    binding = build(
        directory / "build", model.parent / "social-media-mesoscopic-models",
        model.parent / "social-media-models", backend,
    )
    manifest_path = directory / "run_manifest.json"
    manifest = {
        "schema": 1, "plan_fingerprint": fingerprint(plan),
        "binary_sha256": binding["binary_sha256"],
        "source_fingerprint": binding["source_fingerprint"],
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("grid audit runtime changed; use a new directory")
    atomic_json(manifest_path, manifest)
    raw = directory / "raw"
    raw.mkdir(exist_ok=True)
    start = time.monotonic()

    def one(task):
        path = raw / (task["id"] + ".msgpack.lz4")
        if path.exists():
            record = read_record(path)
            if record["task"] != task:
                raise ValueError("grid audit raw task differs from plan")
            return task["id"], 0.0, True
        payload = dict(population_request(task))
        payload["kind"] = task["kind"]
        before = time.monotonic()
        result = binary_result(binding["binary"], payload, None, directory)
        write_record(path, task, result)
        elapsed = time.monotonic() - before
        atomic_json(raw / (task["id"] + ".runtime.json"), {"elapsed_seconds": elapsed})
        return task["id"], elapsed, False

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, task): task for task in plan["tasks"]}
        for done, future in enumerate(as_completed(pending), 1):
            try:
                identifier, elapsed, cached = future.result()
                print(f"{done}/{len(pending)} {identifier} {elapsed:.2f}s {'cached' if cached else 'complete'}", flush=True)
            except Exception as error:
                failures.append({"id": pending[future]["id"], "error": repr(error)})
                print(f"{done}/{len(pending)} FAILED {failures[-1]}", flush=True)
    if failures:
        atomic_json(directory / "failures.json", failures)
        raise RuntimeError(f"{len(failures)} grid audit tasks failed")
    atomic_json(directory / "execution.json", {
        "tasks": len(plan["tasks"]), "workers": workers,
        "elapsed_seconds": time.monotonic() - start,
    })


def comparison(micros, pair, *, comparison_bins=81, bootstrap_samples=1000, bootstrap_seed=87231):
    """Conservative mass projection, never a new micro ensemble for a new mesh."""
    result = pair["result"]
    initial_hash = result["initial_hash"]
    if any(r["result"]["initial_hash"] != initial_hash for r in micros):
        raise ValueError("physical initial state differs across micro/mesh records")
    p = pair["task"]["parameters"]
    finals = [r["result"]["frames"][-1] for r in micros]
    prediction = result["pair"]["frames"][-1]
    if any(f["step"] != prediction["step"] for f in finals):
        raise ValueError("grid comparison does not share an observed endpoint")
    masses = np.asarray([f["rho"] for f in finals])
    masses = masses @ rebin_matrix(micros[0]["task"]["parameters"]["bins"], comparison_bins).T
    predicted = rebin_matrix(p["bins"], comparison_bins) @ np.asarray(prediction["rho"])
    if not np.allclose(masses.sum(axis=1), 1, atol=1e-10) or not np.isclose(predicted.sum(), 1, atol=1e-10):
        raise ValueError("conservative projection lost probability mass")
    masses /= masses.sum(axis=1)[:, None]
    predicted /= predicted.sum()
    mean = masses.mean(axis=0)
    dx = 2 / comparison_bins
    w1 = float(np.abs(np.cumsum(mean - predicted)).sum() * dx)
    rng = np.random.default_rng(bootstrap_seed)
    weights = rng.multinomial(len(masses), np.full(len(masses), 1 / len(masses)), bootstrap_samples) / len(masses)
    boots = weights @ masses
    values = np.abs(np.cumsum(boots - predicted, axis=1)).sum(axis=1) * dx
    low, high = np.quantile(values, (0.025, 0.975))
    micro_series = [[(f["step"], f["exact_node_energy"]) for f in r["result"]["frames"]] for r in micros]
    pair_series = [(f["step"], f["node_energy"]) for f in result["pair"]["frames"]]
    times = np.asarray([s for s, _ in pair_series]) * p["dt"]
    if any([s for s, _ in series] != [s for s, _ in pair_series] for series in micro_series):
        raise ValueError("raw energy paths do not share saved times")
    node = np.asarray([[e for _, e in series] for series in micro_series])
    map_node = np.asarray([e for _, e in pair_series])
    scale = (times[-1] - times[0]) * abs(float(node[:, 0].mean()))
    gap = float(np.trapezoid(np.abs(node.mean(axis=0) - map_node), times) / scale)
    initial = pair["task"]["initial"]
    row = {
        "epsilon": p["epsilon"], "initial": initial, "initial_seed": pair["task"]["initial_seed"],
        "initial_hash": initial_hash, "ratio": p["ratio"], "alpha": p["alpha"],
        "population": p["population"], "bins": p["bins"], "quadrature": p["quadrature"],
        "tau": prediction["step"] * p["alpha"] * p["dt"],
        "comparison_bins": comparison_bins, "processes": len(micros),
        "process_seeds": ",".join(str(r["task"]["process_seed"]) for r in micros),
        "mean_rho_w1": w1, "w1_lower": float(low), "w1_upper": float(high),
        "node_gap_of_mean_curve": gap,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
        "uncertainty": "conditional whole-process bootstrap; excludes initial-state and grid uncertainty",
    }
    fields = [
        {"epsilon": p["epsilon"], "initial": initial, "bins": p["bins"], "bin": i,
         "x": -1 + (i + 0.5) * dx, "micro_mean_mass": float(mean[i]), "map_mass": float(predicted[i])}
        for i in range(comparison_bins)
    ]
    return row, fields


def analyze(directory, plan):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest["plan_fingerprint"] != fingerprint(plan):
        raise ValueError("grid audit run/plan mismatch")
    records = []
    hashes = {}
    for task in plan["tasks"]:
        path = directory / "raw" / (task["id"] + ".msgpack.lz4")
        record = read_record(path)
        if record["task"] != task:
            raise ValueError("grid audit task identity differs")
        records.append(record)
        hashes[str(path.relative_to(directory))] = sha(path)
    rows, fields = [], []
    for epsilon in plan["epsilons"]:
        for initial in range(plan["initials"]):
            group = [r for r in records if r["task"]["parameters"]["epsilon"] == epsilon and r["task"]["initial"] == initial]
            micros = sorted([r for r in group if r["task"]["kind"] == "micro_only"], key=lambda r: r["task"]["branch"])
            pairs = [r for r in group if r["task"]["kind"] == "pair_only"]
            if len(micros) != plan["processes_per_initial"] or len(pairs) != 3:
                raise ValueError("incomplete fixed-ensemble group")
            for pair in pairs:
                row, values = comparison(micros, pair, comparison_bins=plan["comparison_bins"])
                rows.append(row)
                fields.extend(values)
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    table(output / "conditional_ensemble_grid.csv", rows)
    table(output / "comparison_fields.csv", fields)
    summary = []
    for epsilon in plan["epsilons"]:
        selected = [r for r in rows if r["epsilon"] == epsilon]
        for initial in range(plan["initials"]):
            by_bins = {r["bins"]: r for r in selected if r["initial"] == initial}
            summary.append({"epsilon": epsilon, "initial": initial,
                            **{f"w1_B{b}": by_bins[b]["mean_rho_w1"] for b in (81, 161, 241)}})
    table(output / "grid_summary.csv", summary)
    atomic_json(output / "manifest.json", {
        **manifest, "raw_files": hashes, "tasks": len(records),
        "micro_trajectories": sum(r["task"]["kind"] == "micro_only" for r in records),
        "map_trajectories": sum(r["task"]["kind"] == "pair_only" for r in records),
        "analysis_source_sha256": sha(Path(__file__)),
        "files": {p.name: sha(p) for p in output.glob("*.csv")},
    })
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "analyze"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=("purego", "accelerate"), default="accelerate")
    args = parser.parse_args()
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    directory = args.directory.resolve()
    plan = design()
    freeze(directory, plan)
    if args.action == "run":
        execute(directory, plan, args.workers, args.backend)
    if args.action in ("run", "analyze"):
        analyze(directory, plan)


if __name__ == "__main__":
    main()
