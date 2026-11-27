"""Canonical SQLite observations for paired studies and their portable bundles."""

from dataclasses import dataclass
from pathlib import Path

from common.io.json import fingerprint
from common.storage.documents import DocumentStore, DocumentWriter, DocumentWrite
from ehk.domain import (ArmObservation, Binding, Endpoint, FullResult, ImportProvenance, Plan, RawAvailability,
                        StageResult, StagedResult, StoredTask, StudyResult, Task, WireRecord, parse_dataclass)

DATABASE = "analysis.sqlite"
TASKS = "ehk-study-tasks-v1"
PENDING = "ehk-study-pending-v1"
ARMS = "ehk-study-arms-v1"
IMPORTS = "ehk-study-imports-v1"
LOCATIONS = "ehk-study-locations-v1"


@dataclass(kw_only=True, slots=True)
class ArmReference(WireRecord):
    arm_key: str


@dataclass(kw_only=True, slots=True)
class StageIndex(WireRecord):
    checkpoint: Endpoint
    arms: dict[str, ArmReference]


@dataclass(kw_only=True, slots=True)
class StagedIndex(WireRecord):
    reference: ArmReference
    stages: dict[str, StageIndex]


@dataclass(kw_only=True, slots=True)
class FullIndex(WireRecord):
    arms: dict[str, ArmReference]


@dataclass(kw_only=True, slots=True)
class TaskIndex(WireRecord):
    schema: int
    task_id: str
    task_fingerprint: str
    plan_fingerprint: str
    result: StagedIndex | FullIndex
    raw: dict[str, RawAvailability]
    raw_complete: bool
    source: ImportProvenance | None
    status: str | None = None


@dataclass(kw_only=True, slots=True)
class ArmIndex(WireRecord):
    schema: int
    plan_fingerprint: str
    task_id: str
    task_fingerprint: str
    stage: str
    arm: str
    observation: ArmObservation
    raw: RawAvailability
    binding: Binding | None
    source: ImportProvenance | None


@dataclass(kw_only=True, slots=True)
class PendingCommit(WireRecord):
    status: str = "committed"


@dataclass(kw_only=True, slots=True)
class DatasetLocation(WireRecord):
    raw_root: str


def _task_key(task: Task, plan: Plan):
    return fingerprint([plan.plan_fingerprint, task.id])


def arm_items(result: StudyResult):
    if isinstance(result, StagedResult):
        yield "reference", "reference", result.reference
        for stage, block in result.stages.items():
            for name, arm in block.arms.items():
                yield stage, name, arm
    else:
        for name, arm in result.arms.items():
            yield "full", name, arm


def raw_availability(directory, arm: ArmObservation):
    if arm.raw_record is None:
        raise ValueError("stored observations require an explicit raw record identity")
    raw = Path(directory) / arm.raw_record
    missing = [name for name in ("metadata.json", "trajectory.json", "events.db") if not (raw / name).is_file()]
    if not any(raw.glob("finished*")):
        missing.append("finished marker")
    trace = arm.exposure_trace
    if trace is not None and not (Path(directory) / trace.path).is_file():
        missing.append(trace.path)
    return RawAvailability(status="complete" if not missing else "missing", missing=tuple(missing))


class StudyStore:
    """Typed task/arm values; JSON document codecs are confined to this I/O adapter."""

    def __init__(self, directory, *, readonly=False):
        self.directory = Path(directory).resolve()
        self.path = self.directory / DATABASE
        self.documents = DocumentStore(self.path, readonly=True) if readonly else DocumentWriter(self.path)

    def get_task(self, task: Task, plan: Plan, *, pending=False) -> StoredTask | None:
        wire = self.documents.get(PENDING if pending else TASKS, _task_key(task, plan))
        if wire is None or (pending and wire["status"] == "committed"):
            return None
        document = parse_dataclass(TaskIndex, wire)
        if (document.schema != 1 or document.task_id != task.id
                or document.plan_fingerprint != plan.plan_fingerprint
                or document.task_fingerprint != fingerprint(task)):
            raise ValueError(f"stored observation differs from frozen task: {task.id}")

        def resolve(reference: ArmReference, stage: str, name: str):
            expected_key = fingerprint([_task_key(task, plan), stage, name])
            if reference.arm_key != expected_key:
                raise ValueError(f"stored arm reference differs from its frozen identity: {task.id}/{stage}/{name}")
            wire_arm = self.documents.get(ARMS, reference.arm_key)
            if wire_arm is None:
                raise ValueError(f"incomplete stored observation: {task.id}")
            arm = parse_dataclass(ArmIndex, wire_arm)
            if (arm.schema != 1 or arm.task_id != task.id or arm.stage != stage or arm.arm != name
                    or arm.plan_fingerprint != plan.plan_fingerprint
                    or arm.task_fingerprint != document.task_fingerprint):
                raise ValueError(f"stored arm identity differs from its frozen task: {task.id}/{stage}/{name}")
            return arm.observation

        index = document.result
        if isinstance(index, StagedIndex):
            result = StagedResult(reference=resolve(index.reference, "reference", "reference"), stages={
                name: StageResult(checkpoint=stage.checkpoint, arms={arm: resolve(ref, name, arm) for arm, ref in stage.arms.items()})
                for name, stage in index.stages.items()
            })
        else:
            result = FullResult(arms={name: resolve(ref, "full", name) for name, ref in index.arms.items()})
        return StoredTask(schema=document.schema, task_id=document.task_id,
                          task_fingerprint=document.task_fingerprint, plan_fingerprint=document.plan_fingerprint,
                          result=result, raw=document.raw, raw_complete=document.raw_complete, source=document.source)

    def put_task(self, task: Task, result: StudyResult, *, plan: Plan, binding: Binding | None = None,
                 source: ImportProvenance | None = None, complete=True):
        source = source or ImportProvenance(raw_root=str(self.directory))
        task_key = _task_key(task, plan)
        rows, availability, references = [], {}, {}
        for stage, name, observation in arm_items(result):
            key = fingerprint([task_key, stage, name])
            available = raw_availability(source.raw_root, observation)
            availability[f"{stage}/{name}"] = available
            rows.append(DocumentWrite(ARMS, key, ArmIndex(
                schema=1, plan_fingerprint=plan.plan_fingerprint, task_id=task.id,
                task_fingerprint=fingerprint(task), stage=stage, arm=name, observation=observation,
                raw=available, binding=binding, source=source,
            )))
            references[(stage, name)] = ArmReference(arm_key=key)
        if isinstance(result, StagedResult):
            index = StagedIndex(reference=references[("reference", "reference")], stages={
                stage: StageIndex(checkpoint=block.checkpoint, arms={name: references[(stage, name)] for name in block.arms})
                for stage, block in result.stages.items()
            })
        else:
            index = FullIndex(arms={name: references[("full", name)] for name in result.arms})
        document = TaskIndex(schema=1, task_id=task.id, task_fingerprint=fingerprint(task),
                             plan_fingerprint=plan.plan_fingerprint, result=index, raw=availability,
                             raw_complete=bool(availability) and all(item.status == "complete" for item in availability.values()),
                             source=source, status=None if complete else "pending")
        if complete:
            rows.extend([DocumentWrite(TASKS,task_key,document), DocumentWrite(PENDING,task_key,PendingCommit())])
        else:
            rows.append(DocumentWrite(PENDING,task_key,document))
        self.documents.put_many(rows)

    def close(self):
        self.documents.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def task_digest(directory, task: Task, plan: Plan):
    with StudyStore(directory, readonly=True) as store:
        document = store.get_task(task, plan)
    if document is None:
        raise ValueError(f"missing stored observation: {task.id}")
    return fingerprint(document)


def dataset_raw_root(directory, plan: Plan):
    roots = set()
    with StudyStore(directory, readonly=True) as store:
        wire = store.documents.get(LOCATIONS, plan.plan_fingerprint)
        if wire is not None:
            return Path(parse_dataclass(DatasetLocation, wire).raw_root).resolve()
        for task in plan.tasks:
            wire_task = store.documents.get(TASKS, _task_key(task, plan))
            if wire_task is not None:
                source = parse_dataclass(TaskIndex, wire_task).source
                if source is not None:
                    roots.add(source.raw_root)
    if len(roots) > 1:
        raise ValueError("selected observations refer to multiple raw roots; select one dataset")
    return Path(next(iter(roots))).resolve() if roots else Path(directory).resolve()


def read_table(directory, name):
    """Read a named analysis table from its canonical document database."""
    import pandas as pd

    with DocumentStore(Path(directory) / DATABASE, readonly=True) as store:
        document = store.get("tables", name)
    if document is None:
        raise ValueError(f"missing analysis table: {name}")
    return pd.DataFrame(document["rows"], columns=document["columns"])


def write_table(directory, name, frame, *, writer=None):
    """Commit one table; CSV creation is an explicit downstream export."""
    document = {"columns": list(frame.columns), "rows": frame.to_dict("records")}
    if writer is not None:
        writer.put("tables", name, document)
    else:
        with DocumentWriter(Path(directory) / DATABASE) as store:
            store.put("tables", name, document)
