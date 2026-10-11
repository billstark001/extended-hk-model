"""Validate frozen plans and raw records without importing or running a solver."""

import json

from common.io.json import fingerprint, read_result


def records(family, label, outputs):
    """Read a family raw-record list in filename order, skipping hidden AppleDouble sidecars."""
    root = outputs.family(family, label) / "raw"
    # External macOS volumes may store AppleDouble sidecars beside result files.
    return [read_result(p) for p in sorted(root.glob("*.json.gz")) if not p.name.startswith(".")]


def key(meta, exclude=()):
    """Hash comparable metadata, excluding smoke and caller-declared fields."""
    return fingerprint({k: v for k, v in meta.items() if k not in set(exclude) | {"smoke"}})


def validate(root):
    """Return (plan, binding, jobs_by_id) after checking frozen content hash, binding and unique job IDs."""
    plan = json.loads((root / "plan.json").read_text())
    binding = json.loads((root / "binding.json").read_text())
    if fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan["plan_fingerprint"]:
        raise ValueError("frozen plan content/hash mismatch")
    if binding["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("plan/binding mismatch")
    expected = {j["id"]: j for j in plan["jobs"]}
    if len(expected) != len(plan["jobs"]):
        raise ValueError("duplicate frozen job IDs")
    return plan, binding, expected


def checked_records(outputs, label, family, binding, expected):
    """Read raw records and require exact frozen jobs, unique IDs and valid task/build identities; allow a partially completed design."""
    items = records(family, label, outputs)
    if len({r["job"]["id"] for r in items}) != len(items):
        raise ValueError("duplicate raw result identities")
    for r in items:
        job = r["job"]
        if job["id"] not in expected or job != expected[job["id"]]:
            raise ValueError("raw job differs from the frozen plan")
        if r["build_fingerprint"] != binding["source_fingerprint"] or r["identity"] != fingerprint(
            {"task": job, "build": binding["source_fingerprint"]}
        ):
            raise ValueError("raw result/build binding mismatch")
    return items
