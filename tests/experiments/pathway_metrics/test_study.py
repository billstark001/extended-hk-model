"""Frozen preparation and receipt-based recovery, without scientific execution."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from common.storage.documents import DocumentStore
from common.io.json import atomic_json, fingerprint, sha
from experiments.paper.pathway_metrics import study
from experiments.paper.pathway_metrics import collection as core
from ehk.micro.scenarios import create_scenario_metadata


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("read-only workflow stage attempted scientific computation")

    monkeypatch.setattr(core, "run_probe_jobs", forbidden)
    monkeypatch.setattr(core, "analyze", forbidden)
    raw = tmp_path / "raw" / "one"
    raw.mkdir(parents=True)
    for step in (0, 16):
        (raw / f"graph-{step}.msgpack").write_bytes(b"fixed graph identity")
    probe = tmp_path / "probe.exe"
    probe.write_bytes(b"fixed probe identity; never executed")
    primary = tmp_path / "primary.sqlite"
    identity = core.RunIdentity("batch", "one")
    reference = core.PrimaryReference(
        identity, "analysis", str(primary), core.source_identity(raw).fingerprint,
        str(raw), create_scenario_metadata("one"), "fixture", "unavailable", 16,
        .5, True, core.METRIC_REVISION, .8, .9, .7, 2,
    )
    current = [reference]
    monkeypatch.setattr(core, "load_primary", lambda *args, **kwargs: list(current))
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({"schema": 1, "tasks": [{"origin": "batch", "name": "one"}]}))
    directory, database = tmp_path / "study", tmp_path / "comparison.sqlite"
    study.prepare(directory, primary, "analysis", selection, database, probe, expected_runs=1)
    return SimpleNamespace(directory=directory, database=database, reference=reference,
                           current=current, raw=raw, probe=probe)


def commit_receipt(prepared, *, protocol_id=None, reference=None):
    manifest = study.load_manifest(prepared.directory)
    protocol_id = protocol_id or manifest.protocol_id
    reference = reference or prepared.reference
    document = core.ComparisonResult(
        1, "complete", protocol_id, reference,
        (core.Observation(0, .2, 0., .4, .4), core.Observation(16, .8, .2, .6, .4)),
        "simultaneous", 16, 16, .1, verified_graph_snapshots=(0, 16),
        event_reader=core.EventReaderEvidence({"events": "fixture", "rewiring_events": "fixture"}, 0, 0, 0, 0., 0.),
    )
    with DocumentStore(prepared.database) as store:
        store.put(core.NAMESPACE, core.result_key(protocol_id, reference), document)
        store.put(core.PLAN_NAMESPACE, manifest.plan_id + ":status",
                  {"state": "failed", "completed": 0, "failed": 1})


def test_actual_receipt_overrides_interrupted_or_stale_status(prepared):
    commit_receipt(prepared)
    actual = study.status(prepared.directory)
    assert (actual["complete"], actual["missing"], actual["failed"], actual["observations"]) == (1, 0, 0, 2)
    assert study.collect(prepared.directory, plan_only=True)["would_measure"] == 0


def test_another_protocol_is_not_counted_as_completed(prepared):
    commit_receipt(prepared, protocol_id="another-scientific-protocol")
    actual = study.status(prepared.directory)
    assert (actual["complete"], actual["missing"], actual["failed"]) == (0, 1, 0)


def test_prepare_status_and_plan_only_never_start_measurement(prepared):
    # The fixture has already prepared the study with both computation entry
    # points replaced by failures. The plan must retain the unmeasured task.
    result = study.collect(prepared.directory, plan_only=True, workers=2)
    assert result["action"] == "plan_only"
    assert result["would_measure"] == result["missing"] == 1
    with pytest.raises(ValueError, match="explicit --resume"):
        study.collect(prepared.directory)


@pytest.mark.parametrize("changed", ["probe", "raw", "primary", "manifest"])
def test_changed_bindings_refuse_resume_before_computation(prepared, changed):
    if changed == "probe":
        prepared.probe.write_bytes(b"different executable")
    elif changed == "raw":
        (prepared.raw / "graph-16.msgpack").write_bytes(b"different graph")
    elif changed == "primary":
        prepared.current[0] = replace(prepared.reference, I_w=.7)
    else:
        path = prepared.directory / "study.json"
        value = json.loads(path.read_text())
        value["tasks"][0]["terminal_step"] = 32
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed|mismatch"):
        study.collect(prepared.directory, resume=True)


@pytest.mark.parametrize("names,reason", [(["outside"], "outside"), (["one", "one"], "duplicate")])
def test_execution_batch_rejects_outside_or_duplicate_tasks(prepared, names, reason):
    selection = prepared.directory.parent / "batch.json"
    selection.write_text(json.dumps({"schema": 1, "tasks": [{"origin": "batch", "name": name} for name in names]}))
    with pytest.raises(ValueError, match=reason):
        study.collect(prepared.directory, plan_only=True, batch_selection=selection)


def test_batch_is_previewed_then_frozen_without_replacing_full_design(prepared, monkeypatch):
    selection = prepared.directory.parent / "batch.json"
    selection.write_text(json.dumps({"schema": 1, "tasks": [{"origin": "batch", "name": "one"}]}))
    manifest_sha = sha(prepared.directory / "study.json")
    planned = study.collect(prepared.directory, plan_only=True, batch_selection=selection)
    assert (planned["planned"], planned["batch"]["selected"], planned["would_measure"]) == (1, 1, 1)
    assert not (prepared.directory / "runtime/batches").exists()
    calls = []
    monkeypatch.setattr(core, "analyze", lambda *args, **kwargs: calls.append((args, kwargs)))
    study.collect(prepared.directory, resume=True, batch_selection=selection)
    assert len(calls) == 1
    frozen = calls[0][0][2]
    assert frozen.is_relative_to(prepared.directory / "runtime/batches")
    binding = json.loads((frozen.parent / "batch.json").read_text())
    assert binding["selection_sha256"] == sha(frozen)
    assert sha(prepared.directory / "study.json") == manifest_sha


def previous_collector_protocol(protocol):
    current_path = str(__import__("pathlib").Path(core.__file__).resolve())
    previous_path = current_path.replace("pathway_metrics", "pathway_index").replace("collection.py", "cutoff_validation.py")
    sources = dict(protocol.source_sha256)
    sources[previous_path] = sources.pop(current_path)
    return replace(protocol, source_sha256=sources)


def test_relocation_rejects_scientific_or_source_changes(prepared):
    current = study.load_manifest(prepared.directory).protocol
    previous = previous_collector_protocol(current)
    study._authenticate_relocation(previous, current)
    with pytest.raises(ValueError, match="beyond collector relocation"):
        study._authenticate_relocation(replace(previous, observation_interval=8), current)
    corrupted = dict(previous.source_sha256)
    key = next(name for name in corrupted if name.endswith("cutoff_validation.py"))
    corrupted[key] = "different-collector-source"
    with pytest.raises(ValueError, match="source bytes changed"):
        study._authenticate_relocation(replace(previous, source_sha256=corrupted), current)


@pytest.mark.parametrize("completed,pending", [(1, 1), (120, 392)])
def test_explicit_relocation_retains_measurements_and_collects_only_unfinished_tasks(prepared, monkeypatch, completed, pending):
    current = study.load_manifest(prepared.directory)
    previous_protocol = previous_collector_protocol(current.protocol)
    previous_id = fingerprint(previous_protocol)
    for index in range(1, completed + pending):
        name = f"run-{index}"
        raw = prepared.raw.parent / name
        raw.mkdir()
        for step in (0, 16):
            (raw / f"graph-{step}.msgpack").write_bytes(f"fixed graph {index}".encode())
        prepared.current.append(replace(prepared.reference, identity=core.RunIdentity("batch", name), raw_directory=str(raw),
                                        source_fingerprint=core.source_identity(raw).fingerprint,
                                        metadata=replace(prepared.reference.metadata, UniqueName=name)))
    previous_tasks = []
    for reference in prepared.current:
        previous_tasks.append(study.PreparedTask(
            reference.identity, core.result_key(previous_id, reference), fingerprint(reference),
            reference.source_fingerprint, reference.raw_directory, 16, 2, (0, 16)))
    source = prepared.directory.parent / "source-study"
    source.mkdir()
    atomic_json(source / "selection.json", {"schema": 1, "tasks": [asdict(item.identity) for item in previous_tasks]})
    value = json.loads((prepared.directory / "study.json").read_text())
    value.update(study_id="source-study", protocol_id=previous_id, protocol=asdict(previous_protocol),
                 selection_sha256=sha(source / "selection.json"), tasks=[asdict(item) for item in previous_tasks])
    value["manifest_sha256"] = fingerprint({key: item for key, item in value.items() if key != "manifest_sha256"})
    atomic_json(source / "study.json", value)
    for reference in prepared.current[:completed]:
        commit_receipt(prepared, protocol_id=previous_id, reference=reference)
    with DocumentStore(prepared.database, readonly=True) as store:
        original = store.get(core.NAMESPACE, previous_tasks[0].result_key)
    destination = prepared.directory.parent / "relocated-study"
    result = study.migrate(source, destination)
    assert (result["complete"], result["missing"], result["adopted_complete"]) == (completed, pending, completed)
    relocated = study.load_manifest(destination)
    assert relocated.tasks[0].result_key == previous_tasks[0].result_key
    assert relocated.tasks[0].result_protocol_id == previous_id
    assert relocated.tasks[completed].result_protocol_id == relocated.protocol_id
    assert relocated.migration.source_manifest_sha256 == sha(source / "study.json")
    assert (destination / relocated.migration.source_manifest_file).read_bytes() == (source / "study.json").read_bytes()
    assert study.collect(destination, plan_only=True)["would_measure"] == pending
    calls = []
    monkeypatch.setattr(core, "analyze", lambda *args, **kwargs: calls.append(args))
    study.collect(destination, resume=True)
    assert len(calls) == 1
    assert core.load_selection(calls[0][2]) == [reference.identity for reference in prepared.current[completed:]]
    commit_receipt(prepared, protocol_id=relocated.protocol_id, reference=prepared.current[completed])
    actual = study.status(destination)
    assert (actual["complete"], actual["missing"]) == (completed + 1, pending - 1)
    with DocumentStore(prepared.database, readonly=True) as store:
        assert store.get(core.NAMESPACE, previous_tasks[0].result_key) == original
    manifest_path = destination / "study.json"
    original_manifest = manifest_path.read_bytes()
    forged = json.loads(original_manifest)
    forged["tasks"][-1]["result_protocol_id"] = "unauthenticated-foreign-protocol"
    forged["manifest_sha256"] = fingerprint({key: item for key, item in forged.items() if key != "manifest_sha256"})
    atomic_json(manifest_path, forged)
    with pytest.raises(ValueError, match="unauthenticated execution protocol"):
        study.status(destination)
    manifest_path.write_bytes(original_manifest)
    (destination / relocated.migration.source_manifest_file).write_bytes(b"altered source certificate")
    with pytest.raises(ValueError, match="certificate content/hash mismatch"):
        study.load_manifest(destination)
