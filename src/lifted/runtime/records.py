"""Authenticate raw records, input dependencies and all referenced checkpoints."""

from pathlib import Path

from common.io.json import sha
from common.io.records import read_record
from common.runtime.failures import ProtocolError


def record_path(root, identifier):
    return Path(root) / "raw" / (identifier + ".msgpack.lz4")


def read(root, task, binding=None):
    record = read_record(record_path(root, task["id"]))
    result = record["result"]
    if record["task"] != task or (binding is not None and result["binding"] != binding):
        raise ProtocolError("lifted task or executable binding changed")
    for relative, digest in result["files"].items():
        path = (Path(root) / relative).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or sha(path) != digest:
            raise ProtocolError(f"missing or changed checkpoint: {relative}")
    if task["source_task"]:
        parent = record_path(root, task["source_task"])
        if not parent.exists() or sha(parent) != result["dependency_sha256"]:
            raise ProtocolError("source record changed")
    return result


def source_reference(root, task, parent_task):
    parent = read(root, parent_task)
    candidates = parent["paths"] if task["kind"].endswith("extend") else parent["states"]
    selected = [
        r
        for r in candidates
        if (
            r.get("replicate") == task["replicates"][0]
            if task["kind"].endswith("extend")
            else r["step"] == task["source_step"]
        )
    ]
    if len(selected) != 1 or not selected[0].get("checkpoint"):
        raise ProtocolError("source checkpoint is absent or ambiguous")
    ref = {**selected[0], "checkpoint": str(Path(root) / selected[0]["checkpoint"])}
    return ref, sha(record_path(root, parent_task["id"]))
