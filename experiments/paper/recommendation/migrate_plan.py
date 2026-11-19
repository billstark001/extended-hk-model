"""Explicitly extend frozen recommendation plans without changing existing runs."""

import argparse
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from common.experiments.lifecycle import session_lock
from common.experiments.plans import freeze_manifest, read_plan, unique_tasks
from common.io.json import atomic_json, fingerprint, read_result, sha
from common.runtime.failures import ProtocolError

from .design import STUDIES
from .study import make_plan


def _candidate(study):
    content = make_plan(study)
    unique_tasks(content["tasks"])
    return {**content, "plan_fingerprint": fingerprint(content)}


def _validate_extension(saved, candidate):
    identity = lambda plan: {
        key: value for key, value in plan.items() if key not in ("tasks", "protocol", "plan_fingerprint")
    }
    if identity(saved) != identity(candidate):
        raise ProtocolError("plan identity or schema changed; only task additions and reordering are allowed")
    old_protocol, new_protocol = saved["protocol"], candidate["protocol"]
    if (
        {key: value for key, value in old_protocol.items() if key != "settings"}
        != {key: value for key, value in new_protocol.items() if key != "settings"}
    ):
        raise ProtocolError("scientific protocol changed outside the settings extension")
    old_settings = [fingerprint(setting) for setting in old_protocol["settings"]]
    new_settings = [fingerprint(setting) for setting in new_protocol["settings"]]
    if len(set(new_settings)) != len(new_settings) or not set(old_settings) <= set(new_settings):
        raise ProtocolError("the new settings must retain every existing parameter point")
    previous, following = unique_tasks(saved["tasks"]), unique_tasks(candidate["tasks"])
    for identifier, task in previous.items():
        if identifier not in following or fingerprint(task) != fingerprint(following[identifier]):
            raise ProtocolError(f"existing task was removed or changed: {identifier}")


def _records(directory, plan):
    tasks = unique_tasks(plan["tasks"])
    hashes = {}
    for path in sorted((directory / "records").glob("*.json.gz")):
        if path.name.startswith("._"):
            continue
        identifier = path.name.removesuffix(".json.gz")
        if identifier not in tasks:
            raise ProtocolError(f"record is outside the saved plan: {path}")
        record = read_result(path)
        if record.get("task_fingerprint") != fingerprint(tasks[identifier]):
            raise ProtocolError(f"record does not match the saved task: {path}")
        hashes[identifier] = sha(path)
    return hashes


def _pending_receipts(directory, candidate):
    history = directory / "runtime" / "plan_history"
    pending = []
    new_hash = candidate["plan_fingerprint"]
    for path in history.glob("*-to-" + new_hash + ".json"):
        if path.name.startswith("._"):
            continue
        receipt = json.loads(path.read_text())
        if receipt.get("state") != "prepared":
            continue
        old_hash = receipt["old_plan_fingerprint"]
        if path.name != old_hash + "-to-" + new_hash + ".json" or receipt["new_plan_fingerprint"] != new_hash:
            raise ProtocolError(f"invalid prepared migration receipt: {path}")
        saved = read_plan(history / (old_hash + ".json"))
        if saved["plan_fingerprint"] != old_hash:
            raise ProtocolError(f"migration backup/hash mismatch: {path}")
        _validate_extension(saved, candidate)
        pending.append((path, receipt))
    return pending


def _prepare(directory):
    saved = read_plan(directory / "plan.json")
    if saved.get("study") != "paper_recommendation" or saved.get("profile") not in STUDIES:
        raise ProtocolError(f"not a recommendation dataset: {directory}")
    candidate = _candidate(saved["profile"])
    _validate_extension(saved, candidate)
    hashes = _records(directory, saved)
    old_ids = {task["id"] for task in saved["tasks"]}
    return {
        "directory": directory,
        "saved": saved,
        "candidate": candidate,
        "pending_receipts": _pending_receipts(directory, candidate),
        "receipt": {
            "study": saved["profile"],
            "old_plan_fingerprint": saved["plan_fingerprint"],
            "new_plan_fingerprint": candidate["plan_fingerprint"],
            "old_tasks": len(saved["tasks"]),
            "new_tasks": len(candidate["tasks"]),
            "preserved_records": len(hashes),
            "record_sha256": hashes,
            "added_ids": [task["id"] for task in candidate["tasks"] if task["id"] not in old_ids],
        },
    }


def _check_timing_evidence(root, prepared):
    staged = next((item for item in prepared if item["saved"]["profile"] == "staged"), None)
    if staged is None or staged["saved"] == staged["candidate"]:
        return
    binding_path = root / "staged_timing" / "runtime" / "binding.json"
    if binding_path.is_file() and json.loads(binding_path.read_text()).get("stage_evidence") is not None:
        raise ProtocolError("timing already binds staged evidence; its parent plan cannot be extended in place")


def _commit(item):
    directory, saved, candidate = item["directory"], item["saved"], item["candidate"]
    if saved == candidate:
        for path, receipt in item["pending_receipts"]:
            atomic_json(path, {**receipt, "state": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"directory": str(directory), "state": "unchanged", **item["receipt"]}
    history = directory / "runtime" / "plan_history"
    freeze_manifest(history / (saved["plan_fingerprint"] + ".json"), saved)
    receipt_path = history / (saved["plan_fingerprint"] + "-to-" + candidate["plan_fingerprint"] + ".json")
    receipt = {**item["receipt"], "state": "prepared", "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(receipt_path, receipt)
    atomic_json(directory / "plan.json", candidate)
    receipt["state"] = "complete"
    atomic_json(receipt_path, receipt)
    return {"directory": str(directory), "state": "updated", "migration_receipt": str(receipt_path), **item["receipt"]}


def migrate_plans(directory):
    """Validate every existing study before atomically replacing any frozen plan.

    Completed records, raw microscopic histories, jobs and binary bindings remain
    untouched. Missing studies are left for the ordinary runner to initialize.
    Both campaign and study locks must be available, so this command cannot
    update a plan while its experiment is running.
    """
    directory = Path(directory).expanduser().resolve()
    single = (directory / "plan.json").is_file()
    root = directory.parent if single else directory
    selected = [directory] if single else [root / study for study in STUDIES]
    existing = [path for path in selected if (path / "plan.json").is_file()]
    if not existing:
        return [{"directory": str(path), "state": "missing"} for path in selected]
    with ExitStack() as stack:
        campaign = root / "recommendation_runtime"
        if not single or campaign.exists():
            stack.enter_context(session_lock(campaign))
        for path in existing:
            stack.enter_context(session_lock(path / "runtime"))
        prepared = [_prepare(path) for path in existing]
        _check_timing_evidence(root, prepared)
        result = [_commit(item) for item in prepared]
    return result + [{"directory": str(path), "state": "missing"} for path in selected if path not in existing]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Workspace root or one recommendation study directory.")
    args = parser.parse_args(argv)
    summaries = [
        {**{key: value for key, value in item.items() if key not in ("record_sha256", "added_ids")},
         "added_tasks": len(item.get("added_ids", []))}
        for item in migrate_plans(args.directory)
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
