"""Explicitly extend frozen recommendation plans without changing existing runs."""

import argparse
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from common.experiments.lifecycle import session_lock
from common.experiments.plans import freeze_manifest, read_plan, unique_tasks
from common.io.json import atomic_json, fingerprint, read_result, sha
from common.runtime.failures import ProtocolError

from . import runtime

# Historical wire profiles, independent of the current study registries.
STUDIES = ("staged", "staged_timing", "recommender_variants")
IMPLEMENTATIONS = "recommender_variants"


def make_plan(study):
    if study == IMPLEMENTATIONS:
        from ..recommender_variants.study import make_plan as implementations

        return implementations()
    from ..recommendation.study import make_plan as interventions

    return interventions(study)


def _candidate(study):
    content = make_plan(study)
    unique_tasks(content["tasks"])
    return {**content, "plan_fingerprint": fingerprint(content)}


def _validate_extension(saved, candidate):
    def identity(plan):
        return {
            key: value for key, value in plan.items() if key not in ("tasks", "protocol", "plan_fingerprint")
        }

    if identity(saved) != identity(candidate):
        raise ProtocolError("plan identity or schema changed; only additive design extensions are allowed")
    old_protocol, new_protocol = saved["protocol"], candidate["protocol"]
    extended = {"settings"}
    if saved["profile"] == "recommender_variants":
        extended.add("variant_arms")
    if (
        {key: value for key, value in old_protocol.items() if key not in extended}
        != {key: value for key, value in new_protocol.items() if key not in extended}
    ):
        raise ProtocolError("scientific protocol changed outside the additive settings or arm extension")
    if "variant_arms" in old_protocol or "variant_arms" in new_protocol:
        if saved["profile"] != "recommender_variants" or "variant_arms" not in new_protocol:
            raise ProtocolError("variant arm registry cannot be removed or added to another study")
        from ..recommender_variants import design as variant_design

        old_arms, new_arms = variant_design.variant_arms(old_protocol), variant_design.variant_arms(new_protocol)
        for name, spec in old_arms.items():
            if new_arms.get(name) != spec:
                raise ProtocolError(f"existing variant arm was removed or changed: {name}")
        for name, spec in new_arms.items():
            if (
                not isinstance(spec, dict) or set(spec) != {"recommender", "parameters"}
                or not isinstance(spec["recommender"], str) or not isinstance(spec["parameters"], dict)
            ):
                raise ProtocolError(f"invalid variant arm configuration: {name}")
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


def _receipt_name(receipt):
    name = receipt["old_plan_fingerprint"] + "-to-" + receipt["new_plan_fingerprint"]
    if receipt.get("new_binding") is not None and receipt.get("old_binding") != receipt["new_binding"]:
        name += "-binding-" + fingerprint(receipt["new_binding"])
    return name + ".json"


def _pending_receipts(directory, candidate):
    history = directory / "runtime" / "plan_history"
    pending = []
    new_hash = candidate["plan_fingerprint"]
    for path in history.glob("*-to-" + new_hash + "*.json"):
        if path.name.startswith("._"):
            continue
        receipt = json.loads(path.read_text())
        if receipt.get("state") != "prepared":
            continue
        old_hash = receipt["old_plan_fingerprint"]
        if path.name != _receipt_name(receipt) or receipt["new_plan_fingerprint"] != new_hash:
            raise ProtocolError(f"invalid prepared migration receipt: {path}")
        saved = read_plan(history / (old_hash + ".json"))
        if saved["plan_fingerprint"] != old_hash:
            raise ProtocolError(f"migration backup/hash mismatch: {path}")
        _validate_extension(saved, candidate)
        previous_binding = receipt.get("old_binding")
        if previous_binding is not None:
            backup = history / "bindings" / (fingerprint(previous_binding) + ".json")
            if not backup.is_file() or json.loads(backup.read_text()) != previous_binding:
                raise ProtocolError(f"migration binding backup/hash mismatch: {path}")
        pending.append((path, receipt))
    return pending


def _validate_retained_arms(directory, plan, binding):
    """A binary replacement can retain only complete, physically paired arms."""
    from ..recommender_variants import design as variant_design

    names = variant_design.variant_arms(plan["protocol"])
    hashes = binding.get("arm_binary_sha256", dict.fromkeys(names, binding["binary_sha256"]))
    for task in plan["tasks"]:
        record_path = directory / "records" / (task["id"] + ".json.gz")
        if not record_path.is_file():
            raise ProtocolError(f"binary replacement requires every retained task to be complete: {task['id']}")
        arms = read_result(record_path)["result"].get("arms", {})
        initial = None
        for name in names:
            if name not in arms or not runtime.complete_raw(directory, arms[name]):
                raise ProtocolError(f"binary replacement requires complete retained raw arms: {task['id']}/{name}")
            if arms[name].get("binary_sha256", hashes[name]) != hashes[name]:
                raise ProtocolError(f"retained arm binary provenance disagrees with its binding: {task['id']}/{name}")
            checkpoint = runtime.checkpoint_path(directory, task, "full", name, 0)
            if not checkpoint.is_file():
                raise ProtocolError(f"retained arm initial checkpoint is missing: {checkpoint}")
            state = runtime.physical_checkpoint(checkpoint)
            if initial is not None and state != initial:
                raise ProtocolError(f"retained arm initial physical state or RNG differs: {task['id']}/{name}")
            initial = state


def _prepare_binding(directory, saved, candidate, pending, binary_sha256):
    path = directory / "runtime" / "binding.json"
    binding = json.loads(path.read_text()) if path.is_file() else None
    transitions = [receipt for _, receipt in pending if receipt.get("new_binding") is not None]
    if transitions:
        proposed = transitions[0]["new_binding"]
        if any(receipt["new_binding"] != proposed for receipt in transitions):
            raise ProtocolError("multiple pending binary migrations disagree")
        if any(binding not in (receipt["old_binding"], proposed) for receipt in transitions):
            raise ProtocolError("current binding does not match the prepared binary migration")
        if binary_sha256 is not None and proposed["binary_sha256"] != binary_sha256:
            raise ProtocolError("requested binary differs from the prepared migration")
        return binding, proposed
    if saved["profile"] != "recommender_variants":
        if binary_sha256 is not None:
            raise ProtocolError("binary replacement is supported only for recommender_variants")
        return binding, binding
    if binding is None:
        if binary_sha256 is not None:
            raise ProtocolError("binary replacement needs the existing study binding")
        return None, None
    current = binding["binary_sha256"]
    target = binary_sha256 or current
    from ..recommender_variants import design as variant_design

    previous = variant_design.variant_arms(saved["protocol"])
    if binary_sha256 is None and set(variant_design.variant_arms(candidate["protocol"])) - set(previous):
        raise ProtocolError("extending a bound variants study requires an explicit --smp-binary migration")
    old_hashes = binding.get("arm_binary_sha256", dict.fromkeys(previous, current))
    if set(old_hashes) != set(previous):
        raise ProtocolError("existing arm binary provenance does not match the saved protocol")
    if target != current:
        _validate_retained_arms(directory, saved, binding)
    proposed = {
        "binary_sha256": target, "stage_evidence": binding["stage_evidence"],
        "arm_binary_sha256": {
            name: old_hashes.get(name, target) for name in variant_design.variant_arms(candidate["protocol"])
        },
    }
    return binding, proposed


def _prepare(directory, binary_sha256=None):
    saved = read_plan(directory / "plan.json")
    if saved.get("study") != "paper_recommendation" or saved.get("profile") not in STUDIES:
        raise ProtocolError(f"not a recommendation dataset: {directory}")
    candidate = _candidate(saved["profile"])
    _validate_extension(saved, candidate)
    hashes = _records(directory, saved)
    pending = _pending_receipts(directory, candidate)
    old_binding, new_binding = _prepare_binding(directory, saved, candidate, pending, binary_sha256)
    old_ids = {task["id"] for task in saved["tasks"]}
    added_arms = {}
    if saved["profile"] == "recommender_variants" and "variants" in saved["protocol"]:
        from ..recommender_variants import design as variant_design

        previous = variant_design.variant_arms(saved["protocol"])
        added_arms = {
            name: spec for name, spec in variant_design.variant_arms(candidate["protocol"]).items() if name not in previous
        }
    return {
        "directory": directory,
        "saved": saved,
        "candidate": candidate,
        "pending_receipts": pending,
        "old_binding": old_binding,
        "new_binding": new_binding,
        "receipt": {
            "study": saved["profile"],
            "old_plan_fingerprint": saved["plan_fingerprint"],
            "new_plan_fingerprint": candidate["plan_fingerprint"],
            "old_tasks": len(saved["tasks"]),
            "new_tasks": len(candidate["tasks"]),
            "preserved_records": len(hashes),
            "record_sha256": hashes,
            "added_ids": [task["id"] for task in candidate["tasks"] if task["id"] not in old_ids],
            "added_arms": added_arms,
            **({"old_binding": old_binding, "new_binding": new_binding} if old_binding != new_binding else {}),
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
    binding_path = directory / "runtime" / "binding.json"
    if saved == candidate and item["pending_receipts"]:
        if item["new_binding"] is not None:
            atomic_json(binding_path, item["new_binding"])
        for path, receipt in item["pending_receipts"]:
            atomic_json(path, {**receipt, "state": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        state = "updated" if item["old_binding"] != item["new_binding"] else "unchanged"
        return {"directory": str(directory), "state": state, **item["receipt"]}
    if saved == candidate and item["old_binding"] == item["new_binding"]:
        for path, receipt in item["pending_receipts"]:
            atomic_json(path, {**receipt, "state": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"directory": str(directory), "state": "unchanged", **item["receipt"]}
    history = directory / "runtime" / "plan_history"
    freeze_manifest(history / (saved["plan_fingerprint"] + ".json"), saved)
    if item["old_binding"] is not None and item["old_binding"] != item["new_binding"]:
        freeze_manifest(history / "bindings" / (fingerprint(item["old_binding"]) + ".json"), item["old_binding"])
    receipt_path = history / _receipt_name(item["receipt"])
    receipt = {**item["receipt"], "state": "prepared", "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(receipt_path, receipt)
    atomic_json(directory / "plan.json", candidate)
    if item["new_binding"] is not None and item["old_binding"] != item["new_binding"]:
        atomic_json(binding_path, item["new_binding"])
    receipt["state"] = "complete"
    atomic_json(receipt_path, receipt)
    for path, pending in item["pending_receipts"]:
        if path != receipt_path:
            atomic_json(path, {**pending, "state": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"directory": str(directory), "state": "updated", "migration_receipt": str(receipt_path), **item["receipt"]}


def migrate_plans(directory, *, binary=None):
    """Validate every existing study before atomically replacing any frozen plan.

    Completed records, raw microscopic histories and jobs remain untouched.
    A binary replacement retains explicit per-arm provenance and requires every
    prior arm to be complete. Missing studies are initialized by the runner.
    Both campaign and study locks must be available, so this command cannot
    update a plan while its experiment is running.
    """
    directory = Path(directory).expanduser().resolve()
    single = (directory / "plan.json").is_file()
    binary_sha256 = None
    if binary is not None:
        binary = Path(binary).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ProtocolError(f"replacement SMP binary must be executable: {binary}")
        if not single:
            raise ProtocolError("binary replacement must target one recommender_variants study directory")
        binary_sha256 = sha(binary)
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
        prepared = [_prepare(path, binary_sha256) for path in existing]
        _check_timing_evidence(root, prepared)
        result = [_commit(item) for item in prepared]
    return result + [{"directory": str(path), "state": "missing"} for path in selected if path not in existing]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Workspace root or one recommendation study directory.")
    parser.add_argument("--smp-binary", type=Path, help="Explicitly replace the variants runner binary, preserving old arm provenance.")
    args = parser.parse_args(argv)
    summaries = [
        {**{key: value for key, value in item.items() if key not in ("record_sha256", "added_ids")},
         "added_tasks": len(item.get("added_ids", []))}
        for item in migrate_plans(args.directory, binary=args.smp_binary)
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
