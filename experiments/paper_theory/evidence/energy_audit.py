"""Audit finite-map energy accounting and conserved masses at manuscript settings.

The existing CSV bundle supplies source identities, scalar-path comparisons and
the separate long-window conservation audit. Exact mechanism budgets require
new production trace runs because the main measure records did not save them.
No historical generation code or manuscript files are modified by this module.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from common.experiments.plans import freeze_manifest, freeze_plan
from common.io.json import atomic_json, fingerprint, sha
from common.io.records import read_record, read_transport, write_record
from common.io.tables import table
from common.runtime.failures import ProtocolError
from common.runtime.process import acquire_lock, release_lock, run_binary, run_tasks
from theory.parameters import Parameters
from theory.runtime.measure import measure_request

from ..runtime.build import build
from .bundle import Bundle
from .protocol import KERNELS


def manuscript_source(task, *, strict=True):
    """Select the old main design without rebuilding or relabeling its plan."""
    p = task["parameters"]
    return (
        task["kind"] == "measure"
        and p["alpha"] == 0.02
        and p["bins"] == 161
        and p["degree"] == 15
        and p["recommendations"] == 10
        and (not strict or (p["dt"] == 1 and p["quadrature"] == 33))
    )


def design(bundle, epsilons, ratios, horizon):
    """Choose the completed uniform/random-mixing main trajectories to replay."""
    selected = {}
    for task in bundle.plan["tasks"]:
        p = task["parameters"]
        if (
            manuscript_source(task)
            and "effects" in task["roles"]
            and p["population"] == 500
            and p["epsilon"] in epsilons
            and p["ratio"] in ratios
            and p["horizon"] >= horizon
        ):
            key = (p["epsilon"], p["dynamics"], p["recommender"], p["steepness"], p["ratio"])
            if key in selected:
                raise ProtocolError(f"multiple source trajectories for {key}")
            selected[key] = task
    expected = {
        (epsilon, rule, recommender, steepness, ratio)
        for epsilon in epsilons
        for rule in ("hk", "deffuant")
        for recommender, steepness in KERNELS
        for ratio in ratios
    }
    if set(selected) != expected:
        raise ProtocolError(f"missing source settings: {sorted(expected - selected.keys())}")
    tasks = []
    for key in sorted(selected):
        source = selected[key]
        if source["id"] not in bundle.manifest["records"]:
            raise ProtocolError(f"source is not completed: {source['id']}")
        p = replace(Parameters(**source["parameters"]), horizon=horizon, record_tau=0.02)
        core = {"source_task_id": source["id"], "parameters": p.__dict__}
        identifier = "energy_audit_" + fingerprint(core)[:20]
        request = measure_request(identifier, p, final_state=False)
        # Inspection returns its own energies; unrelated response statistics are
        # disabled only for recording. They do not enter model stepping.
        for name, value in request["observables"].items():
            if isinstance(value, bool):
                request["observables"][name] = False
        payload = {
            "id": identifier,
            "kind": "trace",
            "request": request,
            "physical_dt": p.dt,
            "snapshot_steps": [round(24 / (p.alpha * p.dt))] if horizon >= 24 else [],
            "budgets": True,
        }
        tasks.append({"id": identifier, **core, "payload": payload})
    return {
        "schema": 1,
        "source_plan_fingerprint": bundle.plan["plan_fingerprint"],
        "source_data_manifest_sha256": sha(bundle.root / "data_manifest.json"),
        "scope": "deterministic measure map; uniform equal-cell node masses and random-mixing edges",
        "sampling": "every physical opportunity, including initial and final states; no early stopping",
        "epsilons": list(epsilons),
        "ratios": list(ratios),
        "horizon_tau": horizon,
        "energy_units": "E_star = 1",
        "sign_tolerance": 1e-12,
        "tasks": tasks,
    }


def run_one(task, binary, directory):
    from smp_meso_bindings.kinetic_runner import prepare_kinetic_request

    path = directory / (task["id"] + ".msgpack.lz4")
    if path.exists():
        saved = read_record(path)
        if saved["task"] != task:
            raise ProtocolError("cached audit task changed")
        return
    transport = path.with_suffix(".transport")
    try:
        payload = {**task["payload"], "request": prepare_kinetic_request(task["payload"]["request"])}
        run_binary(binary, json.dumps(payload) + "\n", transport, arguments=("-msgpack",), timeout=600)
        result = read_transport(transport, task["id"])
        write_record(path, task, result)
    finally:
        transport.unlink(missing_ok=True)


def audit_trace(task, result, source, tolerance):
    p, frames = task["parameters"], result["frames"]
    expected_steps = task["payload"]["request"]["steps"]
    steps = np.array([f["step"] for f in frames])
    if not np.array_equal(steps, np.arange(expected_steps + 1)):
        raise ProtocolError("energy audit must record every physical opportunity")
    energy = np.array([f["edge_energy"] for f in frames])
    rewiring = np.array([f["budget_rewire"] for f in frames])
    motion = np.array([f["budget_motion"] for f in frames])
    if not all(np.isfinite(values).all() for values in (energy, rewiring, motion)):
        raise ProtocolError("nonfinite energy or cumulative mechanism budget")
    residual = energy - energy[0] - rewiring - motion
    increments = np.diff(rewiring)
    row = {
        "id": task["id"], "source_task_id": task["source_task_id"], **p,
        "physical_opportunities": expected_steps, "saved_states": len(frames),
        "observed_final_tau": expected_steps * p["alpha"] * p["dt"],
        "max_abs_budget_residual": float(np.max(np.abs(residual))),
        "final_budget_residual": float(residual[-1]),
        "initial_edge_energy": float(energy[0]), "final_edge_energy": float(energy[-1]),
        "final_rewiring_budget": float(rewiring[-1]), "final_motion_budget": float(motion[-1]),
        "max_rewiring_increment": float(np.max(increments)),
        "positive_rewiring_increments": int(np.count_nonzero(increments > tolerance)),
        "max_cumulative_rewiring": float(np.max(rewiring)),
        "positive_cumulative_rewiring": bool(np.any(rewiring > tolerance)),
        "positive_final_motion": bool(motion[-1] > tolerance),
        "positive_motion_at_any_step": bool(np.any(motion > tolerance)),
        "max_node_mass_residual": result["diagnostics"]["max_node_mass_residual"],
        "max_fixed_degree_residual": result["diagnostics"]["max_fixed_degree_residual"],
    }
    source_steps = np.rint(source["time"] / p["dt"]).astype(int)
    valid = (source_steps >= 0) & (source_steps <= expected_steps)
    row["source_prefix_points"] = int(np.count_nonzero(valid))
    for kind in ("node", "edge"):
        observed = np.array([frames[k][kind + "_energy"] for k in source_steps[valid]])
        row["max_" + kind + "_energy_source_prefix_gap"] = float(np.max(np.abs(observed - source[kind + "_energy"][valid])))
    for tau in (24, 30):
        step = round(tau / (p["alpha"] * p["dt"]))
        if step <= expected_steps:
            loss = float(energy[0] - energy[step])
            row[f"edge_loss_tau{tau}"] = loss
            row[f"rewiring_budget_tau{tau}"] = float(rewiring[step])
            row[f"motion_budget_tau{tau}"] = float(motion[step])
            row[f"rewiring_fraction_tau{tau}"] = float(-rewiring[step] / loss) if abs(loss) > tolerance else None
    return row


def summarize(rows, tolerance):
    return {
        "trajectories": len(rows),
        "physical_opportunities": sum(r["physical_opportunities"] for r in rows),
        "saved_states": sum(r["saved_states"] for r in rows),
        "min_final_tau": min(r["observed_final_tau"] for r in rows),
        "max_final_tau": max(r["observed_final_tau"] for r in rows),
        **{key: max(r[key] for r in rows) for key in (
            "max_abs_budget_residual", "max_node_mass_residual", "max_fixed_degree_residual",
            "max_rewiring_increment", "max_cumulative_rewiring",
            "max_node_energy_source_prefix_gap", "max_edge_energy_source_prefix_gap",
        )},
        "positive_rewiring_increments": sum(r["positive_rewiring_increments"] for r in rows),
        "trajectories_positive_cumulative_rewiring": sum(r["positive_cumulative_rewiring"] for r in rows),
        "trajectories_positive_final_motion": sum(r["positive_final_motion"] for r in rows),
        "trajectories_positive_motion_at_any_step": sum(r["positive_motion_at_any_step"] for r in rows),
        "sign_tolerance": tolerance,
    }


def conservation(bundle, epsilons):
    """Use the original runtime's all-step maxima as a separate audit cohort."""
    rows = []
    sources = {p["id"]: p for p in bundle.manifest["provenance"]}
    for task in bundle.plan["tasks"]:
        p = task["parameters"]
        if manuscript_source(task, strict=False) and p["epsilon"] in epsilons:
            d = bundle.manifest["records"][task["id"]]["diagnostics"]
            rows.append({
                "source_task_id": task["id"], **p,
                "strict_main_dt1_Q33": manuscript_source(task),
                "executed_steps": d["executed_steps"],
                "observed_final_tau": d["executed_steps"] * p["alpha"] * p["dt"],
                "max_node_mass_residual": d["max_node_mass_residual"],
                "max_fixed_degree_residual": d["max_fixed_degree_residual"],
                "original_raw_path": sources[task["id"]]["path"],
                "original_raw_sha256": sources[task["id"]]["sha256"],
            })
    return rows


def report(bundle, plan, directory):
    from smp_meso_bindings.kinetic_runner import prepare_kinetic_request

    rows, raw_hashes = [], {}
    for task in plan["tasks"]:
        path = directory / "raw" / (task["id"] + ".msgpack.lz4")
        saved = read_record(path)
        if saved["task"] != task:
            raise ProtocolError("audit record/plan mismatch")
        source = bundle.case(bundle.tasks[task["source_task_id"]], fields=False)
        rows.append(audit_trace(task, saved["result"], source, plan["sign_tolerance"]))
        raw_hashes[str(path.relative_to(directory))] = sha(path)
    mass_rows = conservation(bundle, plan["epsilons"])
    summaries = [{"epsilon": epsilon, **summarize([r for r in rows if r["epsilon"] == epsilon], plan["sign_tolerance"])} for epsilon in plan["epsilons"]]
    mass_summaries = []
    for epsilon in plan["epsilons"]:
        for strict in (False, True):
            group = [r for r in mass_rows if r["epsilon"] == epsilon and (not strict or r["strict_main_dt1_Q33"])]
            mass_summaries.append({
                "epsilon": epsilon, "cohort": "strict_dt1_Q33" if strict else "all_alpha002_B161",
                "trajectories": len(group), "executed_steps": sum(r["executed_steps"] for r in group),
                "min_final_tau": min(r["observed_final_tau"] for r in group),
                "max_final_tau": max(r["observed_final_tau"] for r in group),
                **{key: max(r[key] for r in group) for key in ("max_node_mass_residual", "max_fixed_degree_residual")},
            })
    selected = [r for r in rows if r["dynamics"] == "hk" and r["recommender"] == "opinion_random" and r["steepness"] == 4 and r["ratio"] == 2]
    summary = {
        "schema": 1, "plan_fingerprint": plan["plan_fingerprint"],
        "definitions": {
            "max_abs_budget_residual": "max_m |U_E(E_m)-U_E(E_0)-cumulative_R_m-cumulative_M_m| over every saved physical step, including the initial state",
            "max_node_mass_residual": "max_n |sum_i rho_i^n - 1|, including the initial state and every executed physical step",
            "max_fixed_degree_residual": "max_n max_i |sum_j E_ij^n - k_N rho_i^n|; absolute row residual of unnormalized E, not an L1 sum or a residual of e=E/k_N",
            "R_n": "U_E(E_n + q*dt*J[Z_n]) - U_E(E_n)",
            "M_n": "U_E(T_n(E_n + q*dt*J[Z_n])) - U_E(E_n + q*dt*J[Z_n])",
            "positive_final_motion": "cumulative_M at tau=30 exceeds 1e-12 E_star (at the configured horizon for nondefault runs)",
            "positive_motion_at_any_step": "cumulative_M exceeds 1e-12 E_star at at least one saved physical step",
            "positive_rewiring_increments": "count of physical steps where the difference of successive cumulative_R values exceeds 1e-12 E_star",
            "energy_units": "E_star=1; node/edge arrays are cell masses, without extra dx factors",
        },
        "new_energy_audit": summarize(rows, plan["sign_tolerance"]),
        "new_energy_audit_by_epsilon": summaries,
        "original_main_conservation_by_epsilon": mass_summaries,
        "hk_opinion4_r2": [{k: r.get(k) for k in ("epsilon", "id", "source_task_id", "edge_loss_tau24", "rewiring_budget_tau24", "motion_budget_tau24", "rewiring_fraction_tau24")} for r in selected],
        "interpretation": "Mechanism budgets are from 90 new deterministic trajectories by default. Long-window conservation is a separate original cohort; neither supplies microscopic process replicates.",
    }
    table(directory / "energy_audit_trajectories.csv", rows)
    table(directory / "energy_audit_by_epsilon.csv", summaries)
    table(directory / "main_conservation_trajectories.csv", mass_rows)
    table(directory / "main_conservation_by_epsilon.csv", mass_summaries)
    table(directory / "hk_opinion4_r2_tau24.csv", summary["hk_opinion4_r2"])
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "provenance.json", {
        "plan_fingerprint": plan["plan_fingerprint"],
        "audit_run_directory": str(directory.resolve()),
        "source_bundle": str(bundle.root),
        "source_plan_sha256": sha(bundle.root / "plan.json"),
        "source_data_manifest_sha256": sha(bundle.root / "data_manifest.json"),
        "source_scalar_files_sha256": bundle.used,
        "audit_entrypoint_sha256": sha(Path(__file__)),
        "python_sources_sha256": {
            name: sha(Path(inspect.getfile(function)))
            for name, function in (
                ("energy_audit", report), ("measure_request", measure_request),
                ("prepare_kinetic_request", prepare_kinetic_request),
                ("Bundle", Bundle), ("read_record", read_record),
            )
        },
        "build_manifest_sha256": sha(directory / "build" / "manifest.json"),
        "raw_record_sha256": raw_hashes,
        "output_sha256": {p.name: sha(p) for p in sorted(directory.glob("*.csv"))} | {"summary.json": sha(directory / "summary.json")},
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--meso-root", type=Path)
    parser.add_argument("--micro-root", type=Path)
    parser.add_argument("--backend", choices=("accelerate", "purego"), default="accelerate" if os.uname().sysname == "Darwin" else "purego")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.25, 2, 10])
    parser.add_argument("--horizon", type=float, default=30)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only and (args.meso_root is None or args.micro_root is None):
        parser.error("--meso-root and --micro-root are required for new solver runs")
    if len(set(args.epsilons)) != len(args.epsilons) or len(set(args.ratios)) != len(args.ratios):
        parser.error("epsilon and ratio selections must not contain duplicate values")
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    acquire_lock(directory / ".lock")
    try:
        bundle = Bundle(args.source_bundle)
        plan = freeze_plan(directory / "plan.json", design(bundle, args.epsilons, args.ratios, args.horizon))
        if not args.report_only:
            binding = build(directory / "build", args.meso_root.resolve(), args.micro_root.resolve(), args.backend)
            freeze_manifest(directory / "binding.json", {"binary_sha256": binding["binary_sha256"], "source_fingerprint": binding["source_fingerprint"], "plan_fingerprint": plan["plan_fingerprint"]})
            (directory / "raw").mkdir(exist_ok=True)
            started = time.monotonic()
            completed = []
            def done(task, result):
                completed.append(task["id"])
                print(json.dumps({"completed": len(completed), "planned": len(plan["tasks"]), "elapsed_seconds": round(time.monotonic() - started, 2), "id": task["id"]}), flush=True)
            run_tasks(lambda task: run_one(task, binding["binary"], directory / "raw"), plan["tasks"], args.workers, on_result=done)
        summary = report(bundle, plan, directory)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        release_lock(directory / ".lock")


if __name__ == "__main__":
    main()
