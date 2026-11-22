"""Plan extension preserves prior task identities, records and recovery files."""

import json
from pathlib import Path

import pytest

from common.experiments.lifecycle import session_lock
from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, fingerprint, save_result, sha
from common.runtime.failures import ProtocolError
from experiments.paper.recommendation_shared import migration as migrate_plan
from experiments.paper.recommendation_shared import runtime
from experiments.paper.recommendation_shared.migration import make_plan
from experiments.paper.recommender_variants.design import variant_arms


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


def test_factorial_extension_preserves_all_existing_tasks_and_records(tmp_path):
    plan = make_plan("recommender_variants")
    old_protocol = {key: value for key, value in plan["protocol"].items() if key != "variant_arms"}
    previous = {**plan, "protocol": old_protocol}
    saved = freeze_plan(tmp_path / "plan.json", previous)
    task = previous["tasks"][0]
    path = tmp_path / "records" / (task["id"] + ".json.gz")
    save_result(path, {
        "task_fingerprint": fingerprint(task), "result": {"arms": dict.fromkeys(variant_arms(old_protocol), {})},
    })
    original = sha(path)
    result = migrate_plan.migrate_plans(tmp_path)[0]
    migrated = read_plan(tmp_path / "plan.json")
    assert migrated["tasks"] == saved["tasks"] and len(saved["tasks"]) == 496
    assert result["added_ids"] == [] and result["preserved_records"] == 1
    assert set(result["added_arms"]) == {
        "OpinionRandomZeta1Noise0", "OpinionRandomZeta4Noise0", "OpinionRandomZeta4Noise2",
    }
    assert sha(path) == original
    assert len(variant_arms(old_protocol)) == 6 and len(variant_arms(migrated["protocol"])) == 9
    assert all(migrated["protocol"]["variant_arms"][name] == spec for name, spec in variant_arms(old_protocol).items())


@pytest.mark.parametrize("change", ["parameters", "recommender", "removed_arm", "removed_registry"])
def test_arm_extension_rejects_changes_to_existing_native_jobs(tmp_path, monkeypatch, change):
    saved = make_plan("recommender_variants")
    freeze_plan(tmp_path / "plan.json", saved)
    original = (tmp_path / "plan.json").read_bytes()
    proposed = make_plan("recommender_variants")
    arms = proposed["protocol"]["variant_arms"]
    if change == "parameters":
        arms["OpinionRandom"]["parameters"]["Steepness"] = 4.0
    elif change == "recommender":
        arms["OpinionRandom"]["recommender"] = "StructureRandom"
    elif change == "removed_arm":
        arms.pop("OpinionRandomZeta4Noise0")
    else:
        proposed["protocol"].pop("variant_arms")
    monkeypatch.setattr(migrate_plan, "make_plan", lambda profile: proposed)
    with pytest.raises(ProtocolError, match="arm"):
        migrate_plan.migrate_plans(tmp_path)
    assert (tmp_path / "plan.json").read_bytes() == original


def bound_variants(tmp_path, monkeypatch):
    proposed = make_plan("recommender_variants")
    proposed["tasks"] = proposed["tasks"][:2]
    original = {**proposed, "protocol": {key: value for key, value in proposed["protocol"].items() if key != "variant_arms"}}
    directory = tmp_path / "recommender_variants"
    freeze_plan(directory / "plan.json", original)
    atomic_json(directory / "runtime" / "binding.json", {"binary_sha256": "old-binary", "stage_evidence": None})
    for task in original["tasks"]:
        save_result(directory / "records" / (task["id"] + ".json.gz"), {
            "task_fingerprint": fingerprint(task), "result": {
                "arms": {name: {"raw_record": f"raw/{task['id']}/{name}"} for name in variant_arms(original["protocol"])},
            },
        })
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"paired initialization fixture")
    monkeypatch.setattr(migrate_plan, "make_plan", lambda profile: proposed)
    monkeypatch.setattr(runtime, "complete_raw", lambda directory, arm: True)
    monkeypatch.setattr(runtime, "checkpoint_path", lambda *args: checkpoint)
    monkeypatch.setattr(runtime, "physical_checkpoint", lambda path: {"state": "same", "rng": "same"})
    binary = tmp_path / "new-smp"
    binary.write_bytes(b"new binary fixture")
    binary.chmod(0o755)
    return directory, binary


def test_binary_migration_preserves_raw_records_and_names_each_generating_binary(tmp_path, monkeypatch):
    directory, binary = bound_variants(tmp_path, monkeypatch)
    records = {path: sha(path) for path in (directory / "records").glob("*.json.gz")}
    with pytest.raises(ProtocolError, match="--smp-binary"):
        migrate_plan.migrate_plans(directory)
    changed = migrate_plan.migrate_plans(directory, binary=binary)[0]
    assert changed["state"] == "updated" and all(sha(path) == digest for path, digest in records.items())
    binding = json.loads((directory / "runtime" / "binding.json").read_text())
    assert binding["binary_sha256"] == sha(binary) and binding["stage_evidence"] is None
    assert {name: digest for name, digest in binding["arm_binary_sha256"].items() if name not in changed["added_arms"]} == {
        name: "old-binary" for name in variant_arms({"variants": make_plan("recommender_variants")["protocol"]["variants"]})
    }
    assert all(binding["arm_binary_sha256"][name] == sha(binary) for name in changed["added_arms"])
    receipt = json.loads(Path(changed["migration_receipt"]).read_text())
    assert receipt["old_binding"]["binary_sha256"] == "old-binary" and receipt["new_binding"] == binding
    assert len(list((directory / "runtime" / "plan_history" / "bindings").glob("*.json"))) == 1
    assert migrate_plan.migrate_plans(directory, binary=binary)[0]["state"] == "unchanged"


@pytest.mark.parametrize("missing", ["record", "raw", "checkpoint", "pairing"])
def test_binary_replacement_rejects_incomplete_or_unpaired_retained_results(tmp_path, monkeypatch, missing):
    directory, binary = bound_variants(tmp_path, monkeypatch)
    if missing == "record":
        next((directory / "records").glob("*.json.gz")).unlink()
    elif missing == "raw":
        monkeypatch.setattr(runtime, "complete_raw", lambda *args: False)
    elif missing == "checkpoint":
        monkeypatch.setattr(runtime, "checkpoint_path", lambda *args: tmp_path / "missing")
    else:
        states = iter(range(12))
        monkeypatch.setattr(runtime, "physical_checkpoint", lambda *args: next(states))
    plan = (directory / "plan.json").read_bytes()
    binding = (directory / "runtime" / "binding.json").read_bytes()
    with pytest.raises(ProtocolError):
        migrate_plan.migrate_plans(directory, binary=binary)
    assert (directory / "plan.json").read_bytes() == plan
    assert (directory / "runtime" / "binding.json").read_bytes() == binding


@pytest.mark.parametrize("after_write", ["receipt", "plan", "binding"])
def test_prepared_binary_migration_recovers_both_files_after_interruption(tmp_path, monkeypatch, after_write):
    directory, binary = bound_variants(tmp_path, monkeypatch)
    original = migrate_plan.atomic_json
    interrupted = False

    def failed_write(path, value):
        nonlocal interrupted
        original(path, value)
        selected = (
            (after_write == "receipt" and value.get("state") == "prepared")
            or (after_write == "plan" and Path(path) == directory / "plan.json")
            or (after_write == "binding" and Path(path) == directory / "runtime" / "binding.json")
        )
        if selected and not interrupted:
            interrupted = True
            raise RuntimeError("migration interruption fixture")

    monkeypatch.setattr(migrate_plan, "atomic_json", failed_write)
    with pytest.raises(RuntimeError, match="migration interruption fixture"):
        migrate_plan.migrate_plans(directory, binary=binary)
    monkeypatch.setattr(migrate_plan, "atomic_json", original)
    migrate_plan.migrate_plans(directory, binary=binary)
    binding = json.loads((directory / "runtime" / "binding.json").read_text())
    assert binding["binary_sha256"] == sha(binary) and len(binding["arm_binary_sha256"]) == 9
    assert len(read_plan(directory / "plan.json")["protocol"]["variant_arms"]) == 9
    receipts = (directory / "runtime" / "plan_history").glob("*-to-*.json")
    assert all(json.loads(path.read_text())["state"] == "complete" for path in receipts)
