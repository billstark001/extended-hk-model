"""Independent, local-by-default SMP full-state snapshot task for Figure 1(a)."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import cast

import lz4.frame
import msgpack
import numpy as np

from common.experiments.lifecycle import task_lock
from common.experiments.plans import read_plan
from common.io.json import atomic_json, sha
from common.io.records import read_record, write_record
from common.runtime.failures import ProtocolError
from theory.runtime.population import binary_result, population_request
from theory.types import MicroTask, PopulationRequest

from ..types import MainPlan
from .bundle import csv_rows, write_rows


def default_directory(model_root):
    return Path(model_root) / "outputs" / "paper_theory" / "network_snapshot"


def design(horizon=None) -> MainPlan:
    from .main_design import StudyDesign, StudySettings

    horizon = 120.0 if horizon is None else horizon
    if not np.isfinite(horizon) or horizon <= 0 or not np.isclose(horizon / 0.02, round(horizon / 0.02)):
        raise ValueError("snapshot horizon must be a positive multiple of alpha*dt=.02")
    at = min(24.0, horizon)
    builder = StudyDesign(
        StudySettings(
            horizon=horizon,
            epsilons=(0.4,),
            snapshot_taus=(0.0, at, horizon),
            main_snapshot_tau=at,
            population_horizon=horizon,
            adaptive_stopping=False,
        )
    )
    builder.add(
        "micro",
        replace(builder.base, recommender="opinion_random", steepness=4.0, population=500),
        "network_snapshot",
        initial=0,
        branch=0,
        initial_seed=17000,
        process_seed=24000,
        selection="weighted",
    )
    return builder.freeze("network_snapshot", ())


def verified_run(directory):
    plan = read_plan(directory / "plan.json")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ProtocolError("network snapshot plan changed")
    for relative, digest in manifest["files"].items():
        if sha(directory / relative) != digest:
            raise ProtocolError(f"network snapshot changed: {relative}")
    return plan, manifest


def export(directory, output=None):
    """Export the exact checkpoint's graph/opinions; never sample a graph from histograms."""
    directory = Path(directory)
    plan, run = verified_run(directory)
    task = plan["tasks"][0]
    p = task["parameters"]
    at = plan["analysis_protocol"]["main_snapshot_tau"]
    step = round(at / (p["alpha"] * p["dt"]))
    relative = f"checkpoints/checkpoint-{step:09d}.msgpack.lz4"
    envelope = msgpack.unpackb(
        lz4.frame.decompress((directory / relative).read_bytes()), raw=False, strict_map_key=False
    )
    state = msgpack.unpackb(envelope["Data"], raw=False, strict_map_key=False)
    if envelope["CompletedStep"] != step or state["CurStep"] - 1 != step:
        raise ProtocolError("network checkpoint has the wrong completed step")
    opinions = np.asarray(state["Opinions"], dtype=float)
    graph = state["Graph"]
    n, degree = p["population"], p["degree"]
    if not graph["directed"] or set(graph["nodes"]) != set(range(n)) or opinions.shape != (n,):
        raise ProtocolError("network checkpoint node identities differ")
    edges = sorted((int(u), int(v)) for u, targets in graph["adjacency"].items() for v in targets)
    if (
        not np.isfinite(opinions).all()
        or np.max(np.abs(opinions)) > 1
        or any(u == v or not 0 <= u < n or not 0 <= v < n for u, v in edges)
        or not np.all(np.bincount([u for u, _ in edges], minlength=n) == degree)
    ):
        raise ProtocolError("network checkpoint violates opinion/fixed-degree constraints")
    output = Path(output) if output else directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "network_nodes.csv", ["node", "opinion"], enumerate(opinions))
    write_rows(output / "network_edges.csv.gz", ["source", "target"], edges)
    result = read_record(directory / "micro" / "raw" / (task["id"] + ".msgpack.lz4"))["result"]
    frame = next(f for f in result["frames"] if f["step"] == step)
    bins = p["bins"]
    indices = np.clip(((opinions + 1) * bins / 2).astype(int), 0, bins - 1)
    rho = np.bincount(indices, minlength=bins) / n
    edge = np.zeros((bins, bins))
    for u, v in edges:
        edge[indices[u], indices[v]] += 1 / n
    gaps = {
        "node_mass": float(np.max(np.abs(rho - np.asarray(frame["rho"])))),
        "edge_mass": float(np.max(np.abs(edge.ravel() - np.asarray(frame["edge"])))),
    }
    if max(gaps.values()) > 1e-10:
        raise ProtocolError("full network snapshot differs from the recorded microscopic fields")
    manifest = {
        "schema": 1,
        "profile": "network_snapshot",
        "plan_fingerprint": plan["plan_fingerprint"],
        "task": task,
        "tau": at,
        "step": step,
        "nodes": n,
        "edges": len(edges),
        "source_checkpoint": relative,
        "source_checkpoint_sha256": sha(directory / relative),
        "run_manifest_sha256": sha(directory / "run_manifest.json"),
        "binary_sha256": run["binary_sha256"],
        "initial_hash": result["initial_hash"],
        "physical_projection_maximum_error": gaps,
        "fields": "exact node identities, float64 opinions and directed edges from an SMP full-state checkpoint",
        "files": {name: sha(output / name) for name in ("network_nodes.csv", "network_edges.csv.gz")},
    }
    atomic_json(output / "network_manifest.json", manifest)
    from .bundle import Bundle, export_raw
    from .observations import Tables, representative_fields

    export_raw(directory, output)
    bundle = Bundle(output)
    with Tables(output / "tables") as tables:
        representative_fields([bundle.record(task)], plan["analysis_protocol"], tables)
    report = {"planned": 1, "completed": 1, "missing": []}
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "analysis_manifest.json",
        {
            "schema": 1,
            "plan_fingerprint": plan["plan_fingerprint"],
            "data_manifest_sha256": sha(output / "data_manifest.json"),
            "report_sha256": sha(output / "report.json"),
            "tables": {str(path.relative_to(output)): sha(path) for path in sorted((output / "tables").glob("*"))},
        },
    )
    return output


def read_export(directory):
    """Rendering needs only the portable CSV export, without SMP or its raw checkpoint."""
    directory = Path(directory)
    manifest = json.loads((directory / "network_manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if sha(directory / relative) != digest:
            raise ProtocolError(f"network display input changed: {relative}")
    nodes = [(int(r["node"]), float(r["opinion"])) for r in csv_rows(directory / "network_nodes.csv")]
    edges = [(int(r["source"]), int(r["target"])) for r in csv_rows(directory / "network_edges.csv.gz")]
    return nodes, edges, manifest


def dispatch(args):
    from ..runtime.build import build
    from ..study import plan as freeze

    directory = (args.run_dir or args.work_dir or default_directory(args.model_root)).resolve()
    if args.action == "plan":
        return freeze(directory, "network_snapshot", args.horizon)
    if args.action in ("run", "resume"):
        directory.mkdir(parents=True, exist_ok=True)
        with task_lock(directory):
            candidate = freeze(directory, "network_snapshot", args.horizon, resume=args.action == "resume")
            if not (directory / "run_manifest.json").exists():
                siblings = Path(args.model_root).resolve().parent
                binding = build(
                    directory / "build",
                    args.meso_root or siblings / "social-media-mesoscopic-models",
                    args.micro_root or siblings / "social-media-models",
                    args.backend or ("accelerate" if os.uname().sysname == "Darwin" else "purego"),
                )
                task = cast(MicroTask, candidate["tasks"][0])
                payload = dict(population_request(task))
                payload.update(checkpoint_dir=str(directory / "checkpoints"))
                result = binary_result(binding["binary"], cast(PopulationRequest, payload), None, directory)
                raw = directory / "micro" / "raw" / (task["id"] + ".msgpack.lz4")
                write_record(raw, task, result)
                files = [
                    raw,
                    *(directory / "checkpoints").glob("checkpoint-*.msgpack.lz4"),
                    directory / "build" / "manifest.json",
                ]
                atomic_json(
                    directory / "run_manifest.json",
                    {
                        "schema": 1,
                        "plan_fingerprint": candidate["plan_fingerprint"],
                        "binary_sha256": binding["binary_sha256"],
                        "files": {str(path.relative_to(directory)): sha(path) for path in files},
                    },
                )
            analysis = export(directory, args.output_dir)
        if args.phase is None or "plot" in args.phase:
            from .figures import plot

            plot(analysis, directory / "figures", network_dir=analysis)
        return analysis
    if args.action == "analyze":
        return export(directory, args.output_dir)
    if args.action == "plot":
        from .figures import plot

        analysis = directory if (directory / "network_manifest.json").exists() else directory / "analysis"
        return plot(
            analysis,
            args.output_dir or directory / "figures",
            network_dir=analysis,
            intuition_graph=args.intuition_graph,
        )
    if args.action == "status":
        plan, manifest = verified_run(directory)
        status = {"profile": plan["profile"], "planned": 1, "completed": 1, "files_verified": len(manifest["files"])}
        print(json.dumps(status))
        return status
    raise ValueError("network_snapshot supports plan/run/resume/analyze/plot/status")
