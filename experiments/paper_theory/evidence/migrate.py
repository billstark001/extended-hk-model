"""Offline lossless conversion of a frozen run; never import or execute solvers."""

import json
import shutil
from pathlib import Path
from typing import cast

from common.io.json import atomic_json, fingerprint, sha
from common.io.records import equivalent, read_record, record_path, write_record
from theory.runtime.completed import cached
from theory.runtime.population import pair_identity
from theory.runtime.records import read_micro, scientific_arrays

from ..types import FrozenMainPlan, MigrationIdentity, MigrationManifest


def migrate(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("migration requires a separate destination outside the original run")
    plan = cast(FrozenMainPlan, json.loads((source / "plan.json").read_text()))
    if fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan["plan_fingerprint"]:
        raise ValueError("source frozen plan hash mismatch")
    if not (source / "micro" / "raw").is_dir():
        raise FileNotFoundError("raw records unavailable; figure/CSV exports cannot reconstruct trajectories")
    identity: MigrationIdentity = {"source": str(source), "plan_fingerprint": plan["plan_fingerprint"]}
    marker = destination / "migration_manifest.json"
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text())["identity"] != identity:
            raise ValueError("destination belongs to another conversion")
    destination.mkdir(parents=True, exist_ok=True)
    manifest: MigrationManifest = (
        json.loads(marker.read_text())
        if marker.exists()
        else {
            "schema": 1,
            "identity": identity,
            "state": "converting",
            "records": {},
            "scope": "storage conversion only; original task IDs and source/binary provenance preserved",
        }
    )
    atomic_json(marker, manifest)
    shutil.copyfile(source / "plan.json", destination / "plan.json")
    for kind in ("measure", "micro"):
        (destination / kind).mkdir(exist_ok=True)
        shutil.copyfile(source / kind / "binding.json", destination / kind / "binding.json")
    binding = json.loads((source / "micro" / "binding.json").read_text())
    for task in plan["tasks"]:
        raw = source / task["kind"] / "raw"
        original_path = raw / (task["id"] + ".npz") if task["kind"] == "measure" else record_path(raw, task["id"])
        if not original_path.exists() or not cached(original_path, task, task["kind"]):
            raise ValueError(f"source task missing, truncated or incompatible: {task['id']}")
        original_sha = sha(original_path)
        previous = manifest["records"].get(task["id"])
        if previous and previous["source_sha256"] != original_sha:
            raise ValueError(f"source record changed during migration: {task['id']}")
        target_raw = destination / task["kind"] / "raw"
        target_raw.mkdir(exist_ok=True)
        target = target_raw / (original_path.name if task["kind"] == "measure" else task["id"] + ".msgpack.lz4")
        if previous and target.exists() and sha(target) == previous["sha256"] and cached(target, task, task["kind"]):
            continue
        if task["kind"] == "measure":
            shutil.copyfile(original_path, target)
            if sha(target) != original_sha:
                raise ValueError("NPZ copy differs")
        else:
            original = read_micro(original_path)
            converted = scientific_arrays(original["result"])
            prediction = converted.pop("pair")
            # Identical traces were timed independently in the legacy engine.
            # Preserve each branch's diagnostics while sharing scientific arrays.
            original_diagnostics = dict(prediction["diagnostics"])
            prediction = {
                **prediction,
                "diagnostics": {k: v for k, v in original_diagnostics.items() if k != "elapsed_seconds"},
            }
            pair_meta = pair_identity(task, binding)
            key = fingerprint(pair_meta)
            pair_path = destination / "micro" / "pairs" / (key + ".msgpack.lz4")
            pair_result = {
                "initial_hash": converted["initial_hash"],
                "projection_l1": converted.get("projection_l1"),
                "pair": prediction,
            }
            if pair_path.exists():
                if not equivalent(pair_result, read_record(pair_path)["result"]):
                    raise ValueError("process branches disagree on their deterministic pair")
            else:
                write_record(pair_path, {"id": key, **pair_meta}, pair_result)
            converted["pair_ref"] = {"key": key, "sha256": sha(pair_path)}
            converted["pair_diagnostics"] = original_diagnostics
            write_record(target, task, converted)
            restored = read_micro(target)
            restored["result"].pop("pair_ref")
            restored["result"].pop("pair_diagnostics")
            original_scientific = {
                **original,
                "result": {k: v for k, v in original["result"].items() if k not in ("pair_ref", "pair_diagnostics")},
            }
            if not equivalent(original_scientific, restored):
                raise ValueError(f"conversion changed scientific fields: {task['id']}")
        if not cached(target, task, task["kind"]):
            raise ValueError(f"converted record failed validation: {task['id']}")
        manifest["records"][task["id"]] = {
            "source_path": str(original_path.relative_to(source)),
            "source_sha256": original_sha,
            "path": str(target.relative_to(destination)),
            "sha256": sha(target),
        }
        atomic_json(marker, manifest)
    manifest["state"] = "complete"
    atomic_json(marker, manifest)
    # Derived CSVs are deliberately regenerated by the current offline analyzer.
    print(f"Verified storage conversion: {len(plan['tasks'])} unchanged task IDs at {destination}")
    return manifest
