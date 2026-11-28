"""Freeze, inspect and explicitly resume the saved-trajectory I_w/P/E study.

This workflow preserves the existing comparison documents and frozen collector.
Preparation, status and a collection plan never launch scientific computation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import atomic_json, canonical, fingerprint, sha
from common.storage.documents import DocumentStore
from ehk.domain import parse_dataclass

from . import collection as core


@dataclass(frozen=True)
class PreparedTask:
    identity: core.RunIdentity
    result_key: str
    primary_sha256: str
    source_fingerprint: str
    raw_directory: str
    terminal_step: int
    expected_observations: int
    saved_graph_steps: tuple[int, ...]
    result_protocol_id: str | None = None


@dataclass(frozen=True)
class RelocationEvidence:
    source_study: str
    source_manifest_sha256: str
    source_manifest_file: str
    source_study_id: str
    source_protocol_id: str
    collector_old_path: str
    collector_new_path: str
    collector_sha256: str
    adopted_results: int
    equivalence: str


@dataclass(frozen=True)
class StudyManifest:
    schema: int
    study_id: str
    primary_database: str
    analysis_id: str
    comparison_database: str
    selection_file: str
    selection_sha256: str
    selection_source: str
    selection_source_sha256: str
    protocol_id: str
    protocol: core.ComparisonProtocol
    plan_id: str
    tasks: tuple[PreparedTask, ...]
    adopted_complete_keys: tuple[str, ...]
    manifest_sha256: str
    migration: RelocationEvidence | None = None


def load_manifest(directory) -> StudyManifest:
    """Read the immutable design without probing a run or opening its results."""
    directory = Path(directory).resolve()
    value = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    if fingerprint({key: item for key, item in value.items() if key != "manifest_sha256"}) != expected:
        raise ValueError("frozen study manifest content/hash mismatch")
    manifest = parse_dataclass(StudyManifest, value)
    if manifest.schema != 1:
        raise ValueError("unsupported comparison study schema")
    if manifest.migration is not None:
        certificate = directory / manifest.migration.source_manifest_file
        if (certificate.resolve().parent != directory
                or sha(certificate) != manifest.migration.source_manifest_sha256):
            raise ValueError("original study certificate content/hash mismatch")
        certified = json.loads(certificate.read_text(encoding="utf-8"))
        if (certified.get("study_id") != manifest.migration.source_study_id
                or certified.get("protocol_id") != manifest.migration.source_protocol_id):
            raise ValueError("original study certificate identity mismatch")
    selection = directory / manifest.selection_file
    if selection.resolve().parent != directory or sha(selection) != manifest.selection_sha256:
        raise ValueError("frozen selection content/hash mismatch")
    identities = core.load_selection(selection)
    if identities != [task.identity for task in manifest.tasks]:
        raise ValueError("frozen task identities differ from the selected design")
    return manifest


def validate_binding(manifest: StudyManifest, *, verify_sources=False):
    """Reject changed scientific inputs before resuming the frozen collector."""
    if fingerprint(core.make_protocol(Path(manifest.protocol.probe_binary))) != manifest.protocol_id:
        raise ValueError("collector, reader, measurement core or probe changed; use a new protocol/study")
    allowed_protocols = {manifest.protocol_id}
    if manifest.migration is not None:
        evidence = manifest.migration
        earlier = asdict(manifest.protocol)
        digest = earlier["source_sha256"].pop(evidence.collector_new_path, None)
        if digest != evidence.collector_sha256:
            raise ValueError("collector relocation evidence/source hash mismatch")
        earlier["source_sha256"][evidence.collector_old_path] = digest
        if fingerprint(earlier) != evidence.source_protocol_id:
            raise ValueError("collector relocation does not reconstruct the actual original protocol")
        allowed_protocols.add(evidence.source_protocol_id)
    identities = [task.identity for task in manifest.tasks]
    references = core.load_primary(Path(manifest.primary_database), manifest.analysis_id, identities)
    if len(references) != len(manifest.tasks):
        raise ValueError("primary selection coverage changed")
    for task, reference in zip(manifest.tasks, references, strict=True):
        if (task.result_protocol_id or manifest.protocol_id) not in allowed_protocols:
            raise ValueError(f"task refers to an unauthenticated execution protocol: {task.identity}")
        if (fingerprint(reference) != task.primary_sha256
                or core.result_key(task.result_protocol_id or manifest.protocol_id, reference) != task.result_key):
            raise ValueError(f"primary statistics/reference changed: {task.identity}")
        if verify_sources and core.source_identity(task.raw_directory).fingerprint != task.source_fingerprint:
            raise ValueError(f"raw source identity changed: {task.identity}")
    return references


def _record_state(document, task: PreparedTask, protocol_id: str, cadence: int):
    if document is None:
        return "missing", 0, None
    if (document.get("schema") != 1 or document.get("protocol_id") != protocol_id
            or fingerprint(document.get("primary")) != task.primary_sha256):
        return "invalid", 0, "stored result has a different protocol or primary reference"
    if document.get("state") == "failed":
        return "failed", 0, document.get("error")
    if document.get("state") != "complete":
        return "invalid", 0, "unknown committed result state"
    points = document.get("observations", ())
    expected_steps = [*range(0, task.terminal_step, cadence), task.terminal_step]
    if [point.get("step") for point in points] != expected_steps:
        return "invalid", 0, "committed observation grid is incomplete"
    if any(not isinstance(point.get(key), (int, float)) or not math.isfinite(point[key])
           for point in points for key in ("P", "E", "C_edge", "C_pair")):
        return "invalid", 0, "committed P/E observations are missing or nonfinite"
    if tuple(document.get("verified_graph_snapshots", ())) != task.saved_graph_steps:
        return "invalid", 0, "saved-graph replay verification is incomplete"
    if document.get("event_reader") is None:
        return "invalid", 0, "event-reader provenance is absent"
    return "complete", len(points), None


def _inventory(manifest: StudyManifest, tasks=None):
    counts = {name: 0 for name in ("complete", "failed", "missing", "invalid")}
    observations = 0
    errors = []
    complete_keys = []
    database = Path(manifest.comparison_database)
    store = DocumentStore(database, readonly=True) if database.is_file() else None
    try:
        for task in manifest.tasks if tasks is None else tasks:
            document = store.get(core.NAMESPACE, task.result_key) if store is not None else None
            state, points, error = _record_state(document, task, task.result_protocol_id or manifest.protocol_id,
                                                manifest.protocol.observation_interval)
            counts[state] += 1
            observations += points
            if state == "complete":
                complete_keys.append(task.result_key)
            if error is not None:
                errors.append({"identity": asdict(task.identity), "state": state, "error": error})
    finally:
        if store is not None:
            store.close()
    return counts, observations, errors, tuple(complete_keys)


def status(directory, *, verify_sources=False):
    directory = Path(directory).resolve()
    manifest = load_manifest(directory)
    validate_binding(manifest, verify_sources=verify_sources)
    counts, observations, errors, _ = _inventory(manifest)
    state = ("invalid" if counts["invalid"] else "failed" if counts["failed"] else
             "complete" if counts["complete"] == len(manifest.tasks) else "partial")
    return {"schema": 1, "study_id": manifest.study_id, "protocol_id": manifest.protocol_id,
            "plan_id": manifest.plan_id, "state": state, "planned": len(manifest.tasks),
            **counts, "observations": observations, "adopted_complete": len(manifest.adopted_complete_keys),
            "comparison_database": manifest.comparison_database, "errors": errors,
            "resume_command": [sys.executable, "-m", "experiments.paper.pathway_metrics.study",
                               "collect", "--study", str(directory), "--resume", "--workers", "4"]}


def prepare(directory, database, analysis_id, selection, comparison_database, probe, *, expected_runs=512):
    directory, database, selection, comparison_database, probe = (
        Path(path).resolve() for path in (directory, database, selection, comparison_database, probe))
    if database == comparison_database:
        raise ValueError("comparison results must be separate from primary statistics")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    identities = core.load_selection(selection)
    if len(identities) != expected_runs:
        raise ValueError(f"expected {expected_runs} selected tasks, received {len(identities)}")
    manifest_path = directory / "study.json"
    if manifest_path.exists():
        existing = load_manifest(directory)
        if (existing.primary_database != str(database) or existing.analysis_id != analysis_id
                or existing.comparison_database != str(comparison_database)
                or existing.protocol.probe_binary != str(probe)
                or identities != [task.identity for task in existing.tasks]
                or fingerprint(selected) != fingerprint(json.loads((directory / existing.selection_file).read_text(encoding="utf-8")))):
            raise ValueError("prepared study inputs changed; use a new study directory")
        return status(directory, verify_sources=True)
    references = core.load_primary(database, analysis_id, identities)
    protocol = core.make_protocol(probe)
    protocol_id = fingerprint(protocol)
    tasks = []
    pattern = re.compile(r"graph-(\d+)\.msgpack(?:\.lz4)?$")
    for reference in references:
        if core.source_identity(reference.raw_directory).fingerprint != reference.source_fingerprint:
            raise ValueError(f"raw source differs from primary statistics: {reference.identity}")
        steps = tuple(sorted(int(match.group(1)) for path in Path(reference.raw_directory).iterdir()
                             if (match := pattern.fullmatch(path.name))))
        if not steps or steps[0] != 0:
            raise ValueError(f"raw source has no initial graph: {reference.identity}")
        expected_points = len(range(0, reference.terminal_step, protocol.observation_interval)) + 1
        tasks.append(PreparedTask(reference.identity, core.result_key(protocol_id, reference),
                                 fingerprint(reference), reference.source_fingerprint, reference.raw_directory,
                                 reference.terminal_step, expected_points, steps))
    selection_path = directory / "selection.json"
    freeze_manifest(selection_path, selected)
    design = {"schema": 1, "primary_database": str(database), "analysis_id": analysis_id,
              "comparison_database": str(comparison_database), "selection_file": "selection.json",
              "selection_sha256": sha(selection_path), "selection_source": str(selection),
              "selection_source_sha256": sha(selection), "protocol_id": protocol_id, "protocol": protocol,
              "plan_id": fingerprint((analysis_id, identities, protocol_id)), "tasks": tuple(tasks)}
    study_id = fingerprint(design)
    candidate = StudyManifest(**design, study_id=study_id, adopted_complete_keys=(), manifest_sha256="")
    counts, _, errors, complete_keys = _inventory(candidate)
    if counts["invalid"]:
        raise ValueError(f"existing comparison documents fail authentication: {errors[:3]}")
    content = {**design, "study_id": study_id, "adopted_complete_keys": complete_keys, "migration": None}
    manifest = StudyManifest(**content, manifest_sha256=fingerprint(content))
    freeze_manifest(manifest_path, manifest)
    return status(directory, verify_sources=True)


def _authenticate_relocation(previous: core.ComparisonProtocol, current: core.ComparisonProtocol):
    """Require identical source bytes and only one source-map path change."""
    old = asdict(previous)
    new = asdict(current)
    new_path = str(Path(core.__file__).resolve())
    old_paths = [name for name in old["source_sha256"]
                 if name.replace("\\", "/").endswith("/experiments/paper/pathway_index/cutoff_validation.py")]
    if len(old_paths) != 1 or new_path not in new["source_sha256"]:
        raise ValueError("relocation requires the explicitly recorded former collector source path")
    old_path = old_paths[0]
    digest = old["source_sha256"].pop(old_path)
    if digest != new["source_sha256"][new_path]:
        raise ValueError("collector source bytes changed; path-only migration is not authenticated")
    old["source_sha256"][new_path] = digest
    if canonical(old) != canonical(new):
        raise ValueError("measurement, reader, probe or another protocol field changed beyond collector relocation")
    return old_path, new_path, digest


def migrate(source_directory, directory):
    """Explicitly authenticate relocation, retaining completed measurement keys.

    Scientific collection bytes and every other protocol field must match.
    Committed measurements retain their original execution protocol; unfinished
    tasks use the truthful protocol generated at the collector's new location.
    """
    source_directory, directory = (Path(path).resolve() for path in (source_directory, directory))
    if source_directory == directory:
        raise ValueError("migration requires a separate prepared study directory")
    previous = load_manifest(source_directory)
    if previous.migration is not None:
        raise ValueError("this explicit migration accepts the original frozen study, not a migration chain")
    current = core.make_protocol(Path(previous.protocol.probe_binary))
    protocol_id = fingerprint(current)
    old_path, new_path, digest = _authenticate_relocation(previous.protocol, current)
    identities = [task.identity for task in previous.tasks]
    references = core.load_primary(Path(previous.primary_database), previous.analysis_id, identities)
    if len(references) != len(previous.tasks):
        raise ValueError("primary selection coverage changed")
    for task, reference in zip(previous.tasks, references, strict=True):
        if (fingerprint(reference) != task.primary_sha256
                or core.result_key(previous.protocol_id, reference) != task.result_key):
            raise ValueError(f"primary reference changed before relocation: {task.identity}")
        if core.source_identity(task.raw_directory).fingerprint != task.source_fingerprint:
            raise ValueError(f"raw source changed before relocation: {task.identity}")
    counts, _, errors, complete_keys = _inventory(previous)
    if counts["invalid"]:
        raise ValueError(f"existing measurement receipts fail authentication: {errors[:3]}")
    adopted = set(complete_keys)
    tasks = tuple(replace(task, result_protocol_id=previous.protocol_id)
                  if task.result_key in adopted else
                  replace(task, result_key=core.result_key(protocol_id, reference), result_protocol_id=protocol_id)
                  for task, reference in zip(previous.tasks, references, strict=True))
    certificate_bytes = (source_directory / "study.json").read_bytes()
    certificate = directory / "source-study.json"
    if certificate.exists() and certificate.read_bytes() != certificate_bytes:
        raise ValueError("original study certificate changed; use a new study directory")
    if not certificate.exists():
        certificate.parent.mkdir(parents=True, exist_ok=True)
        certificate.write_bytes(certificate_bytes)
    evidence = RelocationEvidence(
        str(source_directory), sha(source_directory / "study.json"), "source-study.json", previous.study_id,
        previous.protocol_id, old_path, new_path, digest, len(complete_keys),
        "Exact collector source bytes and all scientific/probe/reader protocol fields are unchanged; only the collector source-map path was renamed. Completed observations remain part of this same scientific study and retain their actual original execution identities.",
    )
    frozen_selection = directory / "selection.json"
    freeze_manifest(frozen_selection, json.loads((source_directory / previous.selection_file).read_text(encoding="utf-8")))
    design = {"schema": 1, "primary_database": previous.primary_database, "analysis_id": previous.analysis_id,
              "comparison_database": previous.comparison_database, "selection_file": "selection.json",
              "selection_sha256": sha(frozen_selection), "selection_source": str(source_directory / previous.selection_file),
              "selection_source_sha256": sha(source_directory / previous.selection_file),
              "protocol_id": protocol_id, "protocol": current,
              "plan_id": fingerprint((previous.analysis_id, identities, protocol_id)), "tasks": tasks,
              "migration": evidence, "adopted_complete_keys": complete_keys}
    study_id = fingerprint(design)
    content = {**design, "study_id": study_id}
    manifest = StudyManifest(**content, manifest_sha256=fingerprint(content))
    freeze_manifest(directory / "study.json", manifest)
    return status(directory, verify_sources=True)


def _execution_batch(directory: Path, manifest: StudyManifest, selection: Path, *, freeze=False):
    """Choose an explicit subset of the frozen design without replacing it."""
    selection = Path(selection).resolve()
    identities = core.load_selection(selection)
    indexed = {task.identity: task for task in manifest.tasks}
    outside = [identity for identity in identities if identity not in indexed]
    if outside:
        raise ValueError(f"batch selection contains tasks outside the frozen study: {outside[:3]}")
    value = json.loads(selection.read_text(encoding="utf-8"))
    batch_id = fingerprint((manifest.study_id, value))
    batch_directory = directory / "runtime/batches" / batch_id[:20]
    frozen_selection = batch_directory / "selection.json"
    tasks = tuple(indexed[identity] for identity in identities)
    if freeze:
        freeze_manifest(frozen_selection, value)
        freeze_manifest(batch_directory / "batch.json", {
            "schema": 1, "batch_id": batch_id, "study_id": manifest.study_id,
            "protocol_id": manifest.protocol_id, "selection_sha256": sha(frozen_selection),
            "selection_fingerprint": fingerprint(value), "tasks": identities,
            "result_keys": [task.result_key for task in tasks],
        })
    return frozen_selection, tasks, batch_id


def _batch_summary(manifest: StudyManifest, tasks, batch_id, selection):
    counts, observations, errors, _ = _inventory(manifest, tasks)
    return {"batch_id": batch_id, "selected": len(tasks), **counts,
            "observations": observations, "errors": errors, "selection_file": str(selection)}


def collect(directory, *, resume=False, plan_only=False, workers=4, limit=None, batch_selection=None):
    """Only explicit execution resumes measurement; plan-only is read-only."""
    if not plan_only and not resume:
        raise ValueError("collection requires explicit --resume; use --plan-only to inspect")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    directory = Path(directory).resolve()
    manifest = load_manifest(directory)
    validate_binding(manifest, verify_sources=True)
    before = status(directory)
    if before["invalid"]:
        raise ValueError("invalid committed receipts must be resolved before collection")
    selection_path = directory / manifest.selection_file
    batch = None
    if batch_selection is not None:
        selection_path, batch_tasks, batch_id = _execution_batch(directory, manifest, batch_selection)
        batch = _batch_summary(manifest, batch_tasks, batch_id, selection_path)
    selected_status = batch if batch is not None else before
    selected_tasks = batch_tasks if batch_selection is not None else manifest.tasks
    retained = [task for task in selected_tasks
                if (task.result_protocol_id or manifest.protocol_id) != manifest.protocol_id]
    if retained:
        retained_counts, _, _, _ = _inventory(manifest, retained)
        if retained_counts["complete"] != len(retained):
            raise ValueError("an explicitly adopted measurement receipt is unavailable or invalid; restore that receipt before resuming")
    execution_tasks = [task for task in selected_tasks
                       if (task.result_protocol_id or manifest.protocol_id) == manifest.protocol_id]
    if plan_only:
        return {**before, "action": "plan_only", "workers": workers,
                **({"batch": batch} if batch is not None else {}),
                "would_measure": min(selected_status["missing"] + selected_status["failed"], limit)
                if limit is not None else selected_status["missing"] + selected_status["failed"]}
    with session_lock(directory / "runtime"), interruptible():
        try:
            if batch_selection is not None:
                selection_path, batch_tasks, batch_id = _execution_batch(directory, manifest, batch_selection, freeze=True)
            atomic_json(directory / "runtime/run_status.json", {"action": "collect", "state": "running", "workers": workers})
            if execution_tasks and selected_status["missing"] + selected_status["failed"]:
                if retained:
                    # Existing measurements keep their actual protocol and
                    # keys. Only unfinished current-generation tasks are sent
                    # to the collector; no source receipt is rewritten.
                    execution = {"schema": 1, "study_id": manifest.study_id, "protocol_id": manifest.protocol_id,
                                 "tasks": [asdict(task.identity) for task in execution_tasks]}
                    execution_id = fingerprint(execution)
                    selection_path = directory / "runtime/executions" / execution_id[:20] / "selection.json"
                    freeze_manifest(selection_path, execution)
                core.analyze(manifest.primary_database, manifest.analysis_id, selection_path,
                             manifest.comparison_database, manifest.protocol.probe_binary, workers=workers, limit=limit)
            result = status(directory)
            if batch_selection is not None:
                result["batch"] = _batch_summary(manifest, batch_tasks, batch_id, selection_path)
            atomic_json(directory / "runtime/run_status.json", result)
            return result
        except BaseException as error:
            # Task receipts remain authoritative even if an external stop
            # prevents this small execution-status document from being written.
            atomic_json(directory / "runtime/run_status.json",
                        {"action": "collect", "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         "error": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    migrate_parser = commands.add_parser("migrate", help="Authenticate a byte-identical collector relocation and retain completed measurement keys")
    migrate_parser.add_argument("--from-study", type=Path, required=True)
    migrate_parser.add_argument("--study", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze inputs and adopt authentic existing receipts; no measurement")
    prepare_parser.add_argument("--study", type=Path, required=True)
    prepare_parser.add_argument("--database", type=Path, required=True)
    prepare_parser.add_argument("--analysis-id", required=True)
    prepare_parser.add_argument("--selection", type=Path, required=True)
    prepare_parser.add_argument("--comparison-database", type=Path, required=True)
    prepare_parser.add_argument("--probe", type=Path, required=True)
    prepare_parser.add_argument("--expected-runs", type=int, default=512)
    status_parser = commands.add_parser("status", help="Count exact committed task receipts, not old run-status files")
    status_parser.add_argument("--study", type=Path, required=True)
    status_parser.add_argument("--verify-sources", action="store_true")
    collect_parser = commands.add_parser("collect", help="Inspect a plan or explicitly resume the unchanged collector")
    collect_parser.add_argument("--study", type=Path, required=True)
    mode = collect_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    collect_parser.add_argument("--workers", type=int, default=4)
    collect_parser.add_argument("--limit", type=int)
    collect_parser.add_argument("--batch-selection", type=Path, help="Explicit unique subset of the frozen design; full-study receipts remain shared")
    args = parser.parse_args()
    if args.action == "migrate":
        result = migrate(args.from_study, args.study)
    elif args.action == "prepare":
        result = prepare(args.study, args.database, args.analysis_id, args.selection, args.comparison_database,
                         args.probe, expected_runs=args.expected_runs)
    elif args.action == "status":
        result = status(args.study, verify_sources=args.verify_sources)
    else:
        result = collect(args.study, resume=args.resume, plan_only=args.plan_only, workers=args.workers,
                         limit=args.limit, batch_selection=args.batch_selection)
    print(canonical(result), flush=True)
    if result["invalid"] or result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
