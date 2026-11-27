"""Explicit one-time authentication and typed import of historical observations."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from common.experiments.plans import freeze_manifest
from common.experiments.lifecycle import session_lock
from common.io.json import fingerprint, read_result, sha, canonical
from ehk.domain import (ArmObservation, Binding, ImportProvenance, Plan, RawAvailability, StudyResult, Task,
                        WireRecord, parse_binding, parse_observation, parse_result, read_plan)
from .storage import IMPORTS, StudyStore, arm_items


@dataclass(kw_only=True, slots=True)
class HistoricalManifest:
    files: dict[str, str]
    source_dataset: str | None


@dataclass(kw_only=True, slots=True)
class ImportSource:
    study: Path
    manifest_path: Path | None
    manifest: HistoricalManifest | None
    authenticated: dict[str, str]


@dataclass(kw_only=True, slots=True)
class ImportTask:
    task: Task
    result: StudyResult
    path: Path
    sha256: str


@dataclass(kw_only=True, slots=True)
class ImportReceipt(WireRecord):
    schema: int
    import_id: str
    source: str
    source_manifest: str | None
    plan_fingerprint: str
    files: dict[str, str]
    tasks: int
    raw: dict[str, dict[str, RawAvailability]]
    raw_complete: bool


def _source(source: Path):
    manifest_path = source / "data/manifest.json"
    if manifest_path.is_file():
        root, study = source / "data", source / "data/study"
        wire = json.loads(manifest_path.read_text())
        manifest = HistoricalManifest(files=wire["files"], source_dataset=wire.get("source_dataset"))
    elif (source / "source_manifest.json").is_file():
        manifest_path = source / "source_manifest.json"
        root = study = source / "source"
        wire = json.loads(manifest_path.read_text())
        manifest = HistoricalManifest(files=wire["files"], source_dataset=wire.get("source"))
    else:
        return ImportSource(study=source,manifest_path=None,manifest=None,authenticated={})
    authenticated = {}
    for relative, expected in manifest.files.items():
        path = (root / relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"historical bundle authentication failed: {relative}")
        authenticated[str(path)] = expected
    return ImportSource(study=study,manifest_path=manifest_path,manifest=manifest,authenticated=authenticated)


def _validate_observation(observation: ArmObservation, task: Task, binding: Binding, name: str):
    expected = binding.arm_binary_sha256[name] if binding.arm_binary_sha256 is not None else binding.binary_sha256
    if observation.binary_sha256 is not None and observation.binary_sha256 != expected:
        raise ValueError(f"historical per-arm binding differs: {task.id}/{name}")
    if not expected:
        raise ValueError("historical observations require executable provenance")
    steps = [point.step for point in observation.endpoints]
    if not steps or steps != sorted(set(steps)) or observation.stopped_at != steps[-1] or observation.stop_reason != "horizon":
        raise ValueError(f"incomplete historical observation: {task.id}")
    count = task.metadata.NodeCount
    if count is None or len(observation.final_opinions) != count:
        raise ValueError("historical endpoint opinions have the wrong population size")
    previous = steps[0]
    for point in observation.endpoints:
        if point.relative_step != point.step-steps[0] or point.exposure.updates != (point.step-previous)*count:
            raise ValueError("historical observation grid or update denominator differs")
        previous = point.step
    if observation.raw_record is None:
        raise ValueError("historical task observations require raw_record")
    relative = Path(observation.raw_record)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("raw_record must be relative to the recorded dataset")


def _scientific_observation(value):
    return replace(value,raw_record=None,run_directory=None,binary_sha256=None,source_sha256=None,exposure_trace=None)


def import_dataset(source, output, *, raw_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("import destination must be separate from the historical source")
    origin = _source(source)
    plan = read_plan(origin.study / "plan.json")
    binding_path = origin.study / "runtime/binding.json"
    binding = parse_binding(json.loads(binding_path.read_text()))
    authenticated, manifest_files = origin.authenticated, set(origin.authenticated)
    required = [origin.study / "plan.json",binding_path]
    tasks = {task.id: task for task in plan.tasks}
    loaded = []
    recorded_root = origin.manifest.source_dataset if origin.manifest is not None else None
    effective_raw = Path(raw_root).resolve() if raw_root is not None else Path(recorded_root or origin.study).resolve()
    for path in sorted((origin.study / "records").glob("*.json.gz")):
        if path.name.startswith("._"):
            continue
        identifier = path.name.removesuffix(".json.gz")
        if identifier not in tasks:
            raise ValueError(f"historical record lies outside its plan: {identifier}")
        task, digest = tasks[identifier], sha(path)
        wire = read_result(path)
        if wire["task_fingerprint"] != fingerprint(task):
            raise ValueError(f"historical task fingerprint differs: {identifier}")
        result = parse_result(wire["result"])
        for stage,name,observation in arm_items(result):
            if observation.raw_record is None:
                raise ValueError("historical task observations require raw_record")
            observation.raw_record = observation.raw_record.replace("\\","/")
            if observation.exposure_trace is not None:
                observation.exposure_trace.path = observation.exposure_trace.path.replace("\\","/")
            _validate_observation(observation,task,binding,name)
            old_summary = effective_raw / observation.raw_record / "observations.json"
            if old_summary.is_file():
                recorded = parse_observation(json.loads(old_summary.read_text()))
                if _scientific_observation(recorded) != _scientific_observation(observation):
                    raise ValueError(f"raw and task summaries differ: {identifier}/{stage}/{name}")
                authenticated[str(old_summary)] = sha(old_summary)
        required.append(path)
        if sha(path) != digest:
            raise ValueError(f"historical record changed during authentication: {path}")
        authenticated[str(path)] = digest
        loaded.append(ImportTask(task=task,result=result,path=path,sha256=digest))
    if not loaded:
        raise ValueError("historical source contains no task observations")
    if origin.manifest is not None:
        for path in required:
            if str(path.resolve()) not in manifest_files:
                raise ValueError(f"historical manifest does not authenticate required source: {path}")
    for path in required:
        authenticated.setdefault(str(path),sha(path))
    identity = fingerprint(ImportIdentity(plan=plan.plan_fingerprint,files=authenticated,raw_root=str(effective_raw)))
    output.mkdir(parents=True,exist_ok=True)
    with session_lock(output / "runtime"):
        for name,digest in authenticated.items():
            if sha(name) != digest:
                raise ValueError(f"historical source changed before import commit: {name}")
        freeze_manifest(output / "plan.json",plan)
        freeze_manifest(output / "runtime/binding.json",binding)
        with StudyStore(output) as store:
            for item in loaded:
                existing = store.get_task(item.task,plan)
                if existing is not None and existing.result != item.result:
                    raise ValueError(f"import conflicts with existing observations: {item.task.id}")
                if existing is not None and (existing.source is None or existing.source.import_id != identity):
                    raise ValueError("import source or raw-root mapping changed; use a new destination")
                store.put_task(item.task,item.result,plan=plan,binding=binding,
                    source=ImportProvenance(import_id=identity,path=str(item.path),sha256=item.sha256,raw_root=str(effective_raw)))
            raw_states = {item.task.id: store.get_task(item.task,plan).raw for item in loaded}
            receipt = ImportReceipt(schema=1,import_id=identity,source=str(source),
                source_manifest=str(origin.manifest_path) if origin.manifest_path else None,
                plan_fingerprint=plan.plan_fingerprint,files=authenticated,tasks=len(loaded),raw=raw_states,
                raw_complete=all(value.status=="complete" for states in raw_states.values() for value in states.values()))
            store.documents.put(IMPORTS,identity,receipt)
        return receipt


@dataclass(kw_only=True, slots=True)
class ImportIdentity:
    plan: str
    files: dict[str,str]
    raw_root: str


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-root",type=Path)
    args=parser.parse_args(argv)
    receipt=import_dataset(args.source,args.output,raw_root=args.raw_root)
    print(canonical(ImportSummary(import_id=receipt.import_id,tasks=receipt.tasks,raw_complete=receipt.raw_complete)))


@dataclass(kw_only=True,slots=True)
class ImportSummary:
    import_id: str
    tasks: int
    raw_complete: bool


if __name__=="__main__":
    main()
