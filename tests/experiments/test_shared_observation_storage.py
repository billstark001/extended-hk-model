"""One writer, authenticated import, and raw-independent observation reads."""

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from ehk.domain import Task, Binding, FullResult, freeze_plan
from dataclasses import dataclass, replace
from tests.experiments import recommendation_fixtures as fixtures
from ehk.micro.scenarios import create_scenario_metadata
from common.experiments.lifecycle import session_lock
from common.io.json import atomic_json, fingerprint, save_result, sha
from common.storage.documents import DocumentStore, DocumentWriter, DocumentWrite
from experiments.paper.recommendation_shared.analysis import read_study
from experiments.paper.recommendation_shared.import_json import import_dataset
from experiments.paper.recommendation_shared.storage import StudyStore, ARMS, TASKS


@dataclass
class ArrayValue:
    value: np.ndarray


@dataclass
class CommitFlag:
    committed: bool


@dataclass
class LegacyTaskRecord:
    task_fingerprint: str
    result: FullResult


def fixture_study(directory):
    task=Task(id="one",metadata=replace(create_scenario_metadata("one",max_sim_step=16),NodeCount=2),
              setting="fixture",sensitivity="core",replicate=0)
    plan=freeze_plan(directory/"plan.json",fixtures.plan("staged_timing",tasks=(task,)))
    binding=Binding(binary_sha256="frozen-native")
    atomic_json(directory/"runtime/binding.json",binding)
    points=(fixtures.point(0),fixtures.point(16,counters=replace(fixtures.exposure(),updates=32)))
    observation=replace(fixtures.arm(endpoints=points,opinions=(-.4,.4),raw_record="raw/one/full/hold"),stop_reason="horizon")
    return task,plan,binding,FullResult(arms={"hold":observation})


def test_workers_share_one_acknowledged_transaction_owner(tmp_path):
    path = tmp_path / "analysis.sqlite"
    with DocumentWriter(path) as writer:
        def write(index):
            writer.put_many([DocumentWrite("arrays",str(index),ArrayValue(np.arange(index+1))),
                             DocumentWrite("complete",str(index),CommitFlag(True))])
        with ThreadPoolExecutor(max_workers=6) as workers:
            list(workers.map(write, range(30)))
        assert len(writer.values("complete")) == 30
        writer.backup(tmp_path / "portable.sqlite")
    with DocumentStore(tmp_path / "portable.sqlite", readonly=True) as store:
        np.testing.assert_array_equal(store.get("arrays", "29")["value"], np.arange(30))
    (tmp_path / "portable.sqlite").unlink()  # Windows rejects this if backup leaked a connection.


def test_task_and_arm_documents_survive_writer_restart(tmp_path):
    task, plan, binding, result = fixture_study(tmp_path)
    with StudyStore(tmp_path) as store:
        store.put_task(task, result, plan=plan, binding=binding)
    with StudyStore(tmp_path, readonly=True) as store:
        document = store.get_task(task, plan)
        assert document.result == result
        assert document.raw_complete is False
        assert len(list(store.documents.values(ARMS))) == len(list(store.documents.values(TASKS))) == 1
    assert read_study(tmp_path)[1]["one"] == result
    assert not (tmp_path / "records").exists()


def test_explicit_import_preserves_source_and_marks_missing_raw(tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    task, plan, _, result = fixture_study(source)
    record = source / "records/one.json.gz"
    save_result(record, LegacyTaskRecord(fingerprint(task),result))
    before = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    receipt = import_dataset(source, output)
    assert receipt.tasks == 1 and not receipt.raw_complete
    assert read_study(output)[1] == {"one": result}
    assert {str(path): sha(path) for path in source.rglob("*") if path.is_file()} == before
    assert import_dataset(source, output).import_id == receipt.import_id
    # Normal reads never fall back to an existing historical JSON record.
    with pytest.raises(Exception):
        read_study(source)


def test_import_rejects_manifest_omitting_observation(tmp_path):
    source = tmp_path / "bundle"
    study = source / "data/study"
    task, _, _, result = fixture_study(study)
    save_result(study / "records/one.json.gz", LegacyTaskRecord(fingerprint(task),result))
    atomic_json(source / "data/manifest.json", {"schema": 1, "files": {
        str(path.relative_to(source / "data")): sha(path)
        for path in study.rglob("*.json")
    }})
    with pytest.raises(ValueError, match="does not authenticate"):
        import_dataset(source, tmp_path / "new")


def test_import_rejects_wrong_update_denominator_before_writing(tmp_path):
    task, _, _, result = fixture_study(tmp_path / "old")
    result.arms["hold"].endpoints[-1].exposure.updates=31
    save_result(tmp_path / "old/records/one.json.gz", LegacyTaskRecord(fingerprint(task),result))
    with pytest.raises(ValueError, match="denominator"):
        import_dataset(tmp_path / "old", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_import_requires_exclusive_destination_session(tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    task, _, _, result = fixture_study(source)
    save_result(source / "records/one.json.gz", LegacyTaskRecord(fingerprint(task),result))
    with session_lock(output / "runtime"), pytest.raises(RuntimeError, match="already active"):
        import_dataset(source, output)
    assert not (output / "analysis.sqlite").exists()


def test_explicit_import_canonicalizes_windows_relative_paths(tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    task, _, _, result = fixture_study(source)
    result.arms["hold"].raw_record = "raw\\one\\full\\hold"
    record = source / "records/one.json.gz"
    save_result(record, LegacyTaskRecord(fingerprint(task),result))
    original = sha(record)
    import_dataset(source, output)
    assert read_study(output)[1]["one"].arms["hold"].raw_record == "raw/one/full/hold"
    assert sha(record) == original


def test_bundle_export_and_manifest_validation_share_one_exclusive_session(tmp_path, monkeypatch):
    from experiments.paper.recommendation_shared import bundle

    source, output = tmp_path / "source", tmp_path / "bundle"
    task, plan, _, result = fixture_study(source)
    monkeypatch.setattr(bundle, "read_study", lambda directory: (plan, {task.id: result}))
    with session_lock(output / "runtime"):
        with pytest.raises(RuntimeError, match="already active"):
            bundle.export_bundle(source, output)
        with pytest.raises(RuntimeError, match="already active"):
            bundle.read_bundle_manifest(output)
    assert not (output / "data").exists()
    assert not list(output.glob(".export-*"))


@pytest.mark.parametrize("corruption", ["task_schema", "task_id", "task_plan", "arm_key", "arm_schema", "arm_id", "arm_plan", "arm_stage", "arm_name"])
def test_stored_reference_rejects_another_identity(tmp_path, corruption):
    from ehk.domain import parse_dataclass
    from experiments.paper.recommendation_shared.storage import TaskIndex, ArmIndex, ArmReference, FullIndex

    task, plan, binding, result = fixture_study(tmp_path)
    with StudyStore(tmp_path) as store:
        store.put_task(task, result, plan=plan, binding=binding)
    key = fingerprint([plan.plan_fingerprint, task.id])
    with DocumentStore(tmp_path / "analysis.sqlite") as store:
        index = parse_dataclass(TaskIndex, store.get(TASKS, key))
        arm_key = index.result.arms["hold"].arm_key
        stored_arm = parse_dataclass(ArmIndex, store.get(ARMS, arm_key))
        if corruption == "task_schema":
            index = replace(index, schema=2)
        elif corruption == "task_id":
            index = replace(index, task_id="another-task")
        elif corruption == "task_plan":
            index = replace(index, plan_fingerprint="another-plan")
        elif corruption == "arm_key":
            index = replace(index, result=FullIndex(arms={"hold": ArmReference(arm_key="another-arm")}))
        elif corruption == "arm_schema":
            stored_arm = replace(stored_arm, schema=2)
        elif corruption == "arm_id":
            stored_arm = replace(stored_arm, task_id="another-task")
        elif corruption == "arm_plan":
            stored_arm = replace(stored_arm, plan_fingerprint="another-plan")
        elif corruption == "arm_stage":
            stored_arm = replace(stored_arm, stage="late")
        elif corruption == "arm_name":
            stored_arm = replace(stored_arm, arm="another-arm")
        store.put(TASKS, key, index)
        store.put(ARMS, arm_key, stored_arm)
    with StudyStore(tmp_path, readonly=True) as store, pytest.raises(ValueError, match="frozen"):
        store.get_task(task, plan)
