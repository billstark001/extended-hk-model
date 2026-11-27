"""Authenticate portable evidence and align existing metrics without simulation."""

import csv
import gzip
import json
import math
from pathlib import Path

import pandas as pd

from common.experiments.plans import read_plan
from common.io.json import atomic_json, fingerprint, read_result, sha

SCHEMA = 1
IW_SAMPLING_METHOD = "existing_get_indices_adaptive_0.01_max_interval_512"
PATH_KEYS = ["profile", "task_id", "stage", "arm"]
POINT_KEYS = [*PATH_KEYS, "step"]
PROVENANCE = ["plan_fingerprint", "binary_sha256", "raw_record"]
LEGACY_POINTS = ["I_p", "I_s", "I_h"]
LEGACY_PATHS = ["I_w", "sampling_method", "prefix_start", "trajectory_end", "endpoint_complement"]
RAW_REQUIRED = ["metadata.json", "trajectory.json", "events.db", "observations.json"]
POINT_FIELDS = [*POINT_KEYS, "relative_step", "separation", "excess_concordance", "edge_concordance",
                "pair_concordance", *PROVENANCE, "prefix_reference_raw_record"]
PATH_FIELDS = [*PATH_KEYS, *PROVENANCE, "source_dataset", "raw_directory", "start_step", "end_step",
               "full_prefix_required", "prefix_reference_raw_record", "epsilon", "history", "influence",
               "rewiring", "repost", "sensitivity", "replicate", "seed1", "seed2", "available",
               "metadata_json", "seed_json", "expected_rec_params_json", "missing_files", "terminal_status", "major_components", "terminal_json"]


def _new_output(output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists; select a new destination: {output}")
    return output


def endpoint_mode(manifest):
    """These collectors' earlier schema stored the integral alone, without a flag."""
    value = manifest.get("endpoint_complement", False)
    if type(value) is not bool:
        raise ValueError("endpoint_complement must be a boolean")
    return value


def validate_endpoint_column(frame, endpoint_complement):
    """Keep saved values tied to their actual policy, including historical tables."""
    if "endpoint_complement" not in frame:
        if endpoint_complement and len(frame):
            raise ValueError("historical I_w has no endpoint_complement policy; import with --no-endpoint-complement")
        frame["endpoint_complement"] = endpoint_complement
    values = frame["endpoint_complement"].astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any() or not (values == endpoint_complement).all():
        raise ValueError("I_w endpoint_complement differs from the selected policy")
    frame["endpoint_complement"] = values.astype(bool)
    return frame


def _relative(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes evidence directory: {relative}")
    return path


def _authenticate_files(root, manifest):
    if manifest.get("schema") != SCHEMA or not manifest.get("files"):
        raise ValueError("unsupported or empty evidence manifest")
    for relative, expected in manifest["files"].items():
        path = _relative(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"evidence file/hash mismatch: {path}")


def authenticate_preparation(preparation):
    preparation = Path(preparation).resolve()
    manifest = json.loads((preparation / "preparation.json").read_text())
    _authenticate_files(preparation, manifest)
    return manifest


def _bundle_layout(bundle):
    """Read canonical bundles and the authenticated local timing review layout."""
    if (bundle / "data/manifest.json").is_file():
        manifest_path = bundle / "data/manifest.json"
        data, study = bundle / "data", bundle / "data/study"
        manifest = json.loads(manifest_path.read_text())
    elif (bundle / "source_manifest.json").is_file():
        manifest_path = bundle / "source_manifest.json"
        data = study = bundle / "source"
        saved = json.loads(manifest_path.read_text())
        if "schema" in saved:
            raise ValueError("unexpected review manifest schema")
        manifest = {**saved, "schema": SCHEMA, "source_dataset": saved["source"]}
    else:
        raise ValueError(f"no supported authenticated summary manifest: {bundle}")
    _authenticate_files(data, manifest)
    required = {str((study / name).relative_to(data)) for name in ("plan.json", "runtime/binding.json")}
    if not required <= manifest["files"].keys():
        raise ValueError("portable manifest does not authenticate its plan and binding")
    return data, study, manifest_path, manifest


def _write_csv(path, rows, fields):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _binary(binding, name, arm):
    expected = binding["arm_binary_sha256"].get(name) if "arm_binary_sha256" in binding else binding.get("binary_sha256")
    actual = arm.get("binary_sha256", expected)
    if not expected or actual != expected:
        raise ValueError(f"arm binary provenance mismatch: {name}")
    return actual


def _arms(profile, result):
    if profile == "staged":
        yield "reference", "reference", result["reference"], ""
        for stage, block in result["stages"].items():
            for name, arm in block["arms"].items():
                yield stage, name, arm, result["reference"]["raw_record"]
    else:
        for name, arm in result["arms"].items():
            yield "full", name, arm, ""


def prepare(bundles, output, *, raw_roots=None):
    """Export recorded observations and precise requests for missing old metrics."""
    from .raw import _required_files

    output = _new_output(output)
    raw_roots = raw_roots or {}
    sources, paths, tasks = [], [], []
    seen_tasks, seen_paths = set(), set()
    # Validate every source before creating the destination.
    authenticated = []
    for source in bundles:
        bundle = Path(source).expanduser().resolve()
        data, study, manifest_path, manifest = _bundle_layout(bundle)
        plan = read_plan(study / "plan.json")
        if plan["study"] != "paper_recommendation" or plan["profile"] not in (
            "staged", "staged_timing", "recommender_variants"
        ) or manifest["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("portable bundle and frozen plan identity disagree")
        binding = json.loads((study / "runtime/binding.json").read_text())
        if manifest.get("planned") != len(plan["tasks"]):
            raise ValueError("portable manifest planned count disagrees with frozen tasks")
        authenticated.append((bundle, data, study, manifest_path, manifest, plan, binding))
        for task in plan["tasks"]:
            identity = plan["profile"], task["id"]
            if identity in seen_tasks:
                raise ValueError("duplicate task across bundles; choose one generation per study")
            seen_tasks.add(identity)
    if not authenticated:
        raise ValueError("at least one authenticated summary bundle is required")
    if set(raw_roots) - {plan["profile"] for _, _, _, _, _, plan, _ in authenticated}:
        raise ValueError("raw-root profile is outside the selected studies")
    output.mkdir(parents=True)
    point_count = 0
    with gzip.open(output / "points.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, POINT_FIELDS)
        writer.writeheader()
        for bundle, data, study, manifest_path, manifest, plan, binding in authenticated:
            profile = plan["profile"]
            dataset = Path(raw_roots.get(profile, manifest["source_dataset"])).expanduser().resolve()
            source = {"bundle": str(bundle), "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
                      "plan_fingerprint": plan["plan_fingerprint"], "profile": profile,
                      "planned": len(plan["tasks"]), "available_tasks": 0, "missing_tasks": [],
                      "raw_dataset": str(dataset), "binding": binding}
            for task in plan["tasks"]:
                metadata = task["metadata"]
                task_ids = {"profile": profile, "task_id": task["id"],
                            "plan_fingerprint": plan["plan_fingerprint"]}
                tasks.append({**task_ids, "task_fingerprint": fingerprint(task),
                              "metadata_json": json.dumps(metadata, sort_keys=True)})
                record_path = study / f"records/{task['id']}.json.gz"
                relative = str(record_path.relative_to(data))
                if not record_path.is_file():
                    source["missing_tasks"].append(task["id"])
                    continue
                if relative not in manifest["files"]:
                    raise ValueError(f"record is not authenticated by manifest: {relative}")
                record = read_result(record_path)
                if record["task_fingerprint"] != fingerprint(task):
                    raise ValueError(f"record does not match frozen task: {task['id']}")
                source["available_tasks"] += 1
                for stage, name, arm, prefix in _arms(profile, record["result"]):
                    ids = {**task_ids, "stage": stage, "arm": name,
                           "binary_sha256": _binary(binding, name, arm), "raw_record": arm["raw_record"]}
                    key = tuple(ids[k] for k in PATH_KEYS)
                    if key in seen_paths:
                        raise ValueError(f"duplicate path identity: {key}")
                    seen_paths.add(key)
                    raw = _relative(dataset, arm["raw_record"])
                    missing = _required_files(raw)
                    endpoints = arm["endpoints"]
                    if not endpoints:
                        raise ValueError(f"empty observation path: {key}")
                    steps = [point["step"] for point in endpoints]
                    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
                        raise ValueError(f"non-unique or unordered observation steps: {key}")
                    terminal = arm.get("terminal", {})
                    hk = metadata["HKParams"]
                    expected_parameters = dict(metadata["RecSysParams"])
                    if profile == "recommender_variants":
                        registry = plan["protocol"].get("variant_arms")
                        if registry is not None:
                            expected_parameters.update(registry[name]["parameters"])
                    paths.append({**ids, "source_dataset": str(dataset), "raw_directory": str(raw),
                                  "start_step": steps[0], "end_step": steps[-1],
                                  "full_prefix_required": True, "prefix_reference_raw_record": prefix,
                                  "epsilon": hk["Tolerance"], "history": metadata["PostRetainCount"],
                                  "influence": hk["Influence"], "rewiring": hk["RewiringRate"],
                                  "repost": hk["RepostRate"], "sensitivity": task["sensitivity"],
                                  "replicate": task["replicate"], "seed1": metadata["RNG"]["Seed1"],
                                  "seed2": metadata["RNG"]["Seed2"], "available": not missing,
                                  "metadata_json": json.dumps(metadata, sort_keys=True),
                                  "seed_json": json.dumps(metadata["RNG"], sort_keys=True),
                                  "expected_rec_params_json": json.dumps(expected_parameters, sort_keys=True),
                                  "missing_files": json.dumps(missing), "terminal_status": terminal.get("status"),
                                  "major_components": terminal.get("k_major"),
                                  "terminal_json": json.dumps(terminal, sort_keys=True)})
                    for point in endpoints:
                        state = point["pathway"]
                        values = {name: state[name] for name in (
                            "separation", "excess_concordance", "edge_concordance", "pair_concordance"
                        )}
                        if any(value is None or not math.isfinite(value) for value in values.values()):
                            raise ValueError(f"missing/nonfinite recorded observable: {key}/{point['step']}")
                        if not math.isclose(values["excess_concordance"],
                                            values["edge_concordance"] - values["pair_concordance"], abs_tol=1e-12):
                            raise ValueError("recorded excess concordance disagrees with its components")
                        writer.writerow({**ids, "step": point["step"], "relative_step": point["relative_step"],
                                         **values, "prefix_reference_raw_record": prefix})
                        point_count += 1
            sources.append(source)
    _write_csv(output / "paths.csv", paths, PATH_FIELDS)
    _write_csv(output / "tasks.csv", tasks, ["profile", "task_id", "plan_fingerprint", "task_fingerprint", "metadata_json"])
    _write_csv(output / "legacy_points_template.csv", [], [*POINT_KEYS, *PROVENANCE, *LEGACY_POINTS])
    _write_csv(output / "legacy_paths_template.csv", [], [*PATH_KEYS, *PROVENANCE, *LEGACY_PATHS])
    unavailable = sum(not row["available"] for row in paths)
    report = {"schema": SCHEMA, "status": "pending_legacy_metrics", "sources": sources,
              "points": point_count, "paths": len(paths), "available_raw_paths": len(paths) - unavailable,
              "pending_raw_paths": unavailable, "missing_metrics": [*LEGACY_POINTS, "I_w"],
              "weighting": {"I_h": "mean over positive-out-degree users", "C_edge": "mean over directed edges"},
              "path_requirement": "Existing I_w requires existing adaptive get_indices on complete 0..end history; do not integrate the 16-step export as a substitute.",
              "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output / "preparation.json", report)
    return report


def _read_points(preparation):
    return pd.read_csv(preparation / "points.csv.gz", keep_default_na=False, float_precision="round_trip")


def _validate_import(incoming, expected, keys, metrics, *, paths=False, endpoint_complement=False):
    if paths:
        incoming = validate_endpoint_column(incoming.copy(), endpoint_complement)
    required = [*keys, *PROVENANCE, *metrics]
    missing = set(required) - set(incoming)
    if missing:
        raise ValueError(f"legacy metric table is missing columns: {sorted(missing)}")
    if incoming.duplicated(keys).any():
        raise ValueError("duplicate legacy metric key")
    if "step" in keys:
        numeric = pd.to_numeric(incoming["step"], errors="raise")
        if not ((numeric >= 0) & (numeric % 1 == 0)).all():
            raise ValueError("legacy step must be a nonnegative integer")
        incoming["step"] = numeric.astype(int)
    joined = incoming.merge(expected[[*keys, *PROVENANCE]], on=keys, how="left", suffixes=("", "_expected"), indicator=True, validate="one_to_one")
    if (joined["_merge"] != "both").any():
        raise ValueError("legacy metric keys are outside the prepared observations")
    for field in PROVENANCE:
        if not (joined[field] == joined[field + "_expected"]).all():
            raise ValueError(f"legacy metric provenance mismatch: {field}")
    for field in (["I_w"] if paths else LEGACY_POINTS):
        incoming[field] = pd.to_numeric(incoming[field], errors="raise")
        if not incoming[field].map(math.isfinite).all():
            raise ValueError("nonfinite legacy metric; leave missing rows pending instead")
    if paths and len(incoming):
        if not (pd.to_numeric(incoming["prefix_start"], errors="raise") == 0).all():
            raise ValueError("I_w requires the complete prefix starting at zero")
        ends = incoming.merge(expected[[*keys, "end_step"]], on=keys, validate="one_to_one")
        if not (pd.to_numeric(ends["trajectory_end"], errors="raise") == ends["end_step"]).all():
            raise ValueError("I_w trajectory endpoint disagrees with prepared path")
        if not (incoming["sampling_method"] == IW_SAMPLING_METHOD).all():
            raise ValueError("I_w sampling method must use the existing adaptive calculation")
    return incoming[required]


def collect(preparation, output, *, points=None, paths=None, collection=None, endpoint_complement=True):
    """Import supplied old metrics with strict identities and provenance."""
    preparation = Path(preparation).resolve()
    manifest = authenticate_preparation(preparation)
    endpoint_mode({"endpoint_complement": endpoint_complement})
    output = _new_output(output)
    if collection is not None:
        if points is not None or paths is not None:
            raise ValueError("use either --collection or explicit legacy tables")
        collection = Path(collection).resolve()
        supplied = json.loads((collection / "collection.json").read_text())
        _authenticate_files(collection, supplied)
        if supplied.get("preparation_sha256") != sha(preparation / "preparation.json"):
            raise ValueError("legacy collection belongs to a different preparation")
        if endpoint_mode(supplied) != endpoint_complement:
            raise ValueError("legacy collection endpoint_complement differs from the selected policy")
        points, paths = collection / "legacy_points.csv.gz", collection / "legacy_paths.csv"
    if points is None and paths is None:
        raise ValueError("provide existing legacy tables or a collection directory")
    expected_points, expected_paths = _read_points(preparation), pd.read_csv(preparation / "paths.csv", keep_default_na=False, float_precision="round_trip")
    imported = {}
    for name, source, expected, keys, metrics in (
        ("legacy_points.csv.gz", points, expected_points, POINT_KEYS, LEGACY_POINTS),
        ("legacy_paths.csv", paths, expected_paths, PATH_KEYS, LEGACY_PATHS),
    ):
        frame = pd.read_csv(source, keep_default_na=False, float_precision="round_trip") if source is not None else pd.DataFrame(columns=[*keys, *PROVENANCE, *metrics])
        imported[name] = _validate_import(frame, expected, keys, metrics, paths=name == "legacy_paths.csv",
                                          endpoint_complement=endpoint_complement)
    output.mkdir(parents=True)
    for name, frame in imported.items():
        frame.to_csv(output / name, index=False)
    counts = {"points": len(imported["legacy_points.csv.gz"]), "paths": len(imported["legacy_paths.csv"])}
    complete = counts["points"] == manifest["points"] and counts["paths"] == manifest["paths"]
    report = {"schema": SCHEMA, "status": "complete" if complete else "pending_legacy_metrics",
              "endpoint_complement": endpoint_complement,
              "preparation_sha256": sha(preparation / "preparation.json"), "imported": counts,
              "sources": {name: {"path": str(Path(p).resolve()), "sha256": sha(Path(p))}
                          for name, p in (("points", points), ("paths", paths)) if p is not None},
              "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output / "collection.json", report)
    return report


def compare(preparation, collection, output):
    """Join exact recorded states and report missing coverage; add no classifications."""
    preparation, collection = Path(preparation).resolve(), Path(collection).resolve()
    authenticate_preparation(preparation)
    output = _new_output(output)
    source = json.loads((collection / "collection.json").read_text())
    endpoint_complement = endpoint_mode(source)
    _authenticate_files(collection, source)
    if source.get("preparation_sha256") != sha(preparation / "preparation.json"):
        raise ValueError("legacy collection belongs to a different preparation")
    output.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "complete", "coverage": {}, "endpoint_complement": endpoint_complement}
    for name, expected, keys, metrics in (
        ("points", _read_points(preparation), POINT_KEYS, LEGACY_POINTS),
        ("paths", pd.read_csv(preparation / "paths.csv", keep_default_na=False, float_precision="round_trip"), PATH_KEYS, LEGACY_PATHS),
    ):
        table = collection / ("legacy_points.csv.gz" if name == "points" else "legacy_paths.csv")
        incoming = pd.read_csv(table, keep_default_na=False, float_precision="round_trip")
        incoming = _validate_import(incoming, expected, keys, metrics, paths=name == "paths",
                                    endpoint_complement=endpoint_complement)
        joined = expected.merge(incoming[[*keys, *metrics]], on=keys, how="left", validate="one_to_one", indicator="legacy_coverage")
        joined.to_csv(output / f"{name}_aligned.csv.gz", index=False)
        missing = int((joined["legacy_coverage"] != "both").sum())
        report["coverage"][name] = {"requested": len(expected), "matched": len(incoming), "pending": missing}
        if missing:
            report["status"] = "pending_legacy_metrics"
    report.update(preparation_sha256=sha(preparation / "preparation.json"),
                  collection_sha256=sha(collection / "collection.json"),
                  files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    atomic_json(output / "comparison.json", report)
    return report
