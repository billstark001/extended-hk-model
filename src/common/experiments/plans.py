"""Frozen identities, immutable manifests and coverage independent of record format."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from common.io.json import atomic_json, fingerprint
from common.runtime.failures import ProtocolError
from common.types import TaskCoverage


def unique_tasks(tasks: Iterable[Mapping[str, Any]], *, allow_empty: bool = False) -> dict[str, Any]:
    items = list(tasks)
    indexed = {t["id"]: t for t in items}
    if (not allow_empty and not items) or len(indexed) != len(items):
        raise ProtocolError("empty design or duplicate identifiers")
    return indexed


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    content = {k: v for k, v in plan.items() if k != "plan_fingerprint"}
    if fingerprint(content) != plan.get("plan_fingerprint"):
        raise ProtocolError("frozen plan content/hash mismatch")
    unique_tasks(plan.get("tasks", plan.get("jobs", [])))
    return cast(dict[str, Any], plan)


def freeze_plan(path: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    content = dict(candidate)
    content.pop("plan_fingerprint", None)
    unique_tasks(content.get("tasks", content.get("jobs", [])))
    frozen = {**content, "plan_fingerprint": fingerprint(content)}
    if path.exists():
        if read_plan(path) != frozen:
            raise ProtocolError("experiment plan changed; use a new output directory")
    else:
        atomic_json(path, frozen)
    return frozen


def freeze_manifest(path: Path, candidate: Mapping[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != dict(candidate):
            raise ProtocolError(f"binding or manifest changed: {path}; use a new run")
    else:
        atomic_json(path, candidate)


def write_design(path: Path, tasks: Iterable[Mapping[str, Any]]) -> list[Any]:
    selected = list(unique_tasks(tasks).values())
    candidate = {"schema": 1, "tasks": selected}
    if path.exists() and json.loads(path.read_text()) != candidate:
        raise ProtocolError(f"design changed: {path}; use a new output root")
    if not path.exists():
        atomic_json(path, candidate)
    return selected


def coverage(tasks: Iterable[Mapping[str, Any]], available: Iterable[str]) -> TaskCoverage:
    expected = unique_tasks(tasks)
    found = set(available)
    if found - expected.keys():
        raise ProtocolError("result IDs are outside the frozen plan")
    missing = [identifier for identifier in expected if identifier not in found]
    return {"planned": len(expected), "completed": len(found), "missing": missing}


def dataset_directory(root: Path, container: str = "evidence") -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "plan.json").is_file():
        return root
    candidates = [p.parent for p in (root / container).glob("*/plan.json")]
    if len(candidates) != 1:
        raise ValueError("expected one frozen dataset; specify its directory directly")
    return candidates[0]
