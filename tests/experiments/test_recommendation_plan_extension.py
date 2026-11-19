"""Plan extension preserves prior task identities, records and recovery files."""

import json

import pytest

from common.experiments.lifecycle import session_lock
from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, fingerprint, save_result, sha
from common.runtime.failures import ProtocolError
from experiments.paper.recommendation import migrate_plan


def candidate(profile="staged"):
    return {
        "schema": 3, "study": "paper_recommendation", "profile": profile,
        "protocol": {"settings": [[0.01, 0.3], [0.03, 0.1], [0.01, 0.1]], "cadence": 16, "replicates": 2},
        "tasks": [
            {"id": "a0", "metadata": {"RNG": {"Seed1": "0x1"}}},
            {"id": "b0", "metadata": {"RNG": {"Seed1": "0x2"}}},
            {"id": "new0", "metadata": {"RNG": {"Seed1": "0x3"}}},
            {"id": "a1", "metadata": {"RNG": {"Seed1": "0x4"}}},
        ],
    }


def previous(profile="staged"):
    plan = candidate(profile)
    plan["protocol"]["settings"] = plan["protocol"]["settings"][:2]
    plan["tasks"] = [plan["tasks"][0], plan["tasks"][3], plan["tasks"][1]]
    return plan


def write_previous(root, profile="staged"):
    directory = root / profile
    saved = freeze_plan(directory / "plan.json", previous(profile))
    return directory, saved


def test_extension_preserves_records_jobs_bindings_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate_plan, "make_plan", candidate)
    directory, saved = write_previous(tmp_path)
    record = directory / "records" / "a0.json.gz"
    save_result(record, {"task_fingerprint": fingerprint(saved["tasks"][0]), "result": {"complete": True}})
    job = directory / "raw" / "a0" / "job.json"
    atomic_json(job, {"seed": "0x1", "interrupted_step": 17})
    binding = directory / "runtime" / "binding.json"
    atomic_json(binding, {"binary_sha256": "frozen-binary"})
    original = {path: sha(path) for path in (record, job, binding)}

    result = migrate_plan.migrate_plans(tmp_path)
    updated = read_plan(directory / "plan.json")
    assert updated["tasks"] == candidate()["tasks"]
    assert all(sha(path) == digest for path, digest in original.items())
    assert read_plan(directory / "runtime" / "plan_history" / (saved["plan_fingerprint"] + ".json")) == saved
    entry = next(row for row in result if row["state"] == "updated")
    assert entry["old_tasks"] == 3 and entry["new_tasks"] == 4
    assert entry["preserved_records"] == 1 and entry["added_ids"] == ["new0"]
    assert entry["record_sha256"] == {"a0": original[record]}
    receipt_path = directory / "runtime" / "plan_history" / (
        saved["plan_fingerprint"] + "-to-" + updated["plan_fingerprint"] + ".json"
    )
    assert json.loads(receipt_path.read_text())["state"] == "complete"
    (receipt_path.parent / ("._" + receipt_path.name)).write_bytes(b"\x00\xb0resource fork")
    second = migrate_plan.migrate_plans(tmp_path)
    assert next(row for row in second if row["directory"] == str(directory))["state"] == "unchanged"
    assert not (tmp_path / "recommender_variants").exists()


@pytest.mark.parametrize("change", ["seed", "removed", "protocol", "schema", "settings"])
def test_extension_rejects_changed_science_or_task_identity(tmp_path, monkeypatch, change):
    directory, _ = write_previous(tmp_path)
    original = (directory / "plan.json").read_bytes()
    proposed = candidate()
    if change == "seed":
        proposed["tasks"][0]["metadata"]["RNG"]["Seed1"] = "0x99"
    elif change == "removed":
        proposed["tasks"].pop(0)
    elif change == "protocol":
        proposed["protocol"]["cadence"] = 1
    elif change == "schema":
        proposed["schema"] = 4
    else:
        proposed["protocol"]["settings"].pop(0)
    monkeypatch.setattr(migrate_plan, "make_plan", lambda profile: proposed)
    with pytest.raises(ProtocolError):
        migrate_plan.migrate_plans(tmp_path)
    assert (directory / "plan.json").read_bytes() == original
    assert not (directory / "runtime" / "plan_history").exists()


def test_order_only_migration_keeps_every_task_unchanged(tmp_path, monkeypatch):
    directory, saved = write_previous(tmp_path)
    proposed = previous()
    proposed["tasks"].reverse()
    monkeypatch.setattr(migrate_plan, "make_plan", lambda profile: proposed)
    result = migrate_plan.migrate_plans(directory)
    assert result[0]["state"] == "updated" and result[0]["added_ids"] == []
    updated = read_plan(directory / "plan.json")
    assert {task["id"]: fingerprint(task) for task in updated["tasks"]} == {
        task["id"]: fingerprint(task) for task in saved["tasks"]
    }


def test_multistudy_validation_finishes_before_any_plan_is_replaced(tmp_path, monkeypatch):
    first, _ = write_previous(tmp_path, "staged")
    second, _ = write_previous(tmp_path, "recommender_variants")
    original = {path: (path / "plan.json").read_bytes() for path in (first, second)}

    def invalid_variant(profile):
        proposed = candidate(profile)
        if profile == "recommender_variants":
            proposed["tasks"][0]["metadata"]["RNG"]["Seed1"] = "0x99"
        return proposed

    monkeypatch.setattr(migrate_plan, "make_plan", invalid_variant)
    with pytest.raises(ProtocolError):
        migrate_plan.migrate_plans(tmp_path)
    assert all((path / "plan.json").read_bytes() == content for path, content in original.items())


def test_active_campaign_or_study_cannot_be_migrated(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate_plan, "make_plan", candidate)
    directory, _ = write_previous(tmp_path)
    for control in (tmp_path / "recommendation_runtime", directory / "runtime"):
        with session_lock(control), pytest.raises(RuntimeError, match="already active"):
            migrate_plan.migrate_plans(tmp_path)


def test_bound_timing_evidence_prevents_parent_plan_expansion(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate_plan, "make_plan", candidate)
    directory, saved = write_previous(tmp_path)
    atomic_json(tmp_path / "staged_timing" / "runtime" / "binding.json", {
        "binary_sha256": "frozen-binary", "stage_evidence": {"plan": saved["plan_fingerprint"]},
    })
    original = (directory / "plan.json").read_bytes()
    with pytest.raises(ProtocolError, match="timing already binds"):
        migrate_plan.migrate_plans(tmp_path)
    assert (directory / "plan.json").read_bytes() == original


def test_invalid_record_fingerprint_prevents_plan_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate_plan, "make_plan", candidate)
    directory, _ = write_previous(tmp_path)
    save_result(directory / "records" / "a0.json.gz", {"task_fingerprint": "wrong"})
    original = (directory / "plan.json").read_bytes()
    with pytest.raises(ProtocolError, match="record does not match"):
        migrate_plan.migrate_plans(tmp_path)
    assert (directory / "plan.json").read_bytes() == original


def test_missing_plans_are_not_created(tmp_path):
    result = migrate_plan.migrate_plans(tmp_path)
    assert len(result) == 3 and all(item["state"] == "missing" for item in result)
    assert list(tmp_path.iterdir()) == []


def test_committed_plan_finishes_a_prepared_receipt_on_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate_plan, "make_plan", candidate)
    directory, _ = write_previous(tmp_path)
    migrate_plan.migrate_plans(tmp_path)
    receipt_path = next((directory / "runtime" / "plan_history").glob("*-to-*.json"))
    receipt = json.loads(receipt_path.read_text())
    atomic_json(receipt_path, {**receipt, "state": "prepared"})
    plan_bytes = (directory / "plan.json").read_bytes()
    result = migrate_plan.migrate_plans(tmp_path)
    assert next(row for row in result if row["directory"] == str(directory))["state"] == "unchanged"
    assert json.loads(receipt_path.read_text())["state"] == "complete"
    assert (directory / "plan.json").read_bytes() == plan_bytes
