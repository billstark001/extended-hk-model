"""Frozen plan identities and bound raw-result readers."""

import collections
import json
import os
from datetime import datetime, timezone

from common.io.json import atomic_json, canonical, fingerprint

from .protocol import FAMILIES


def result_path(job, label, outputs):
    """Return the immutable gzip raw-record location for an identified job and run label."""
    return outputs.family(job["family"], label) / "raw" / (job["id"] + ".json.gz")


def summarize(jobs):
    """Return task counts by scientific family and tier, including empty declared families."""
    return {family: dict(collections.Counter(j["tier"] for j in jobs if j["family"] == family)) for family in FAMILIES}


def resolve_frozen_plan(candidate, frozen=None):
    """Keep the original identity when only definition-file bytes changed.

    The complete generated requests, metadata, seeds, profile and analysis
    protocol must remain identical in canonical JSON. Solver source/binary
    checks are independent and stay strict. Original definition hashes remain
    in the frozen plan rather than being replaced by the current file hashes.
    """
    if frozen is None:
        return candidate, fingerprint(candidate), False
    original = {k: v for k, v in frozen.items() if k != "plan_fingerprint"}
    original_id = frozen["plan_fingerprint"]
    if fingerprint(original) != original_id:
        raise RuntimeError("frozen plan content/hash mismatch")
    if fingerprint(candidate) == original_id:
        return original, original_id, False
    ignored = {"definitions_sha256"}
    if fingerprint({k: v for k, v in candidate.items() if k not in ignored}) != fingerprint(
        {k: v for k, v in original.items() if k not in ignored}
    ):
        raise RuntimeError("plan changed; use a new label, never combine incompatible protocols")
    return original, original_id, True


def record_definition_check(root, plan_id, original_hashes, current_hashes):
    """Audit a compatible source edit without changing the plan or binding."""
    path = root / "definition_checks.jsonl"
    check = {
        "plan_fingerprint": plan_id,
        "frozen_definition_sha256": original_hashes,
        "current_definition_sha256": current_hashes,
        "result": "identical_generated_jobs_and_analysis_protocol",
    }
    check_id = fingerprint(check)
    if path.exists() and any(
        json.loads(line)["check_id"] == check_id for line in path.read_text().splitlines() if line.strip()
    ):
        return
    with path.open("a") as f:
        f.write(
            canonical({**check, "check_id": check_id, "checked_at_utc": datetime.now(timezone.utc).isoformat()}) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def write_design(path, tasks):
    """Materialize unique nonempty tasks and freeze design.json; identical designs resume, changed designs require a new output root."""
    tasks = list(tasks)
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("empty design or duplicate identifiers")
    design = {"schema": 1, "tasks": tasks}
    if path.exists():
        if json.loads(path.read_text()) != design:
            raise ValueError(f"design changed: {path}; use a new output root")
    else:
        atomic_json(path, design)
    return tasks


def read_plan(path):
    """Load plan.json and verify its canonical content hash before returning the saved plan."""
    plan = json.loads(path.read_text())
    if fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan["plan_fingerprint"]:
        raise ValueError("frozen plan content/hash mismatch")
    return plan
