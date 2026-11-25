"""Isolated writer-process SIGKILL checks, not power-loss durability tests.

The benchmark coordinator stays alive.  A child reports a successful backend
operation, the coordinator fsyncs its ACK journal, and only then can that child
be killed.  An opt-in adapter hook also permits killing before SQLite COMMIT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import signal
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

BACKENDS = ("current", "tinymongo", "indexed_split", "sqlite_optimized")
SCHEDULE = (
    ("after_ack", 2),
    ("before_commit", 3),
    ("after_ack", 4),
    ("before_commit", 4),
    ("after_ack", 5),
    ("before_commit", 5),
)


class RecoveryFailure(RuntimeError):
    """A recovery invariant or the fault-injection protocol failed."""


def array_manifest(payload: dict) -> dict:
    result = {}

    def walk(value, path):
        if isinstance(value, np.ndarray):
            result[path] = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                part = str(key).replace("~", "~0").replace("/", "~1")
                walk(item, path + "/" + part if path else part)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, path + "/" + str(index))

    walk(payload, "")
    return result


def _canonical(value):
    if isinstance(value, np.ndarray):
        return {"$array": array_manifest({"value": value})["value"]}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("recovery fixtures require finite scalars")
    return value


def document_hash(document: dict) -> str:
    encoded = json.dumps(_canonical(document), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_document(identifier: str, generation: int, payload_bytes: int) -> dict:
    """Finite, deterministic fixtures with independently checkable array metadata."""
    if payload_bytes < 1:
        raise ValueError("payload_bytes must be positive")
    count = max(8, payload_bytes // 12)
    payload = {
        "p_index": np.arange(count, dtype=np.float64) / 8 + generation,
        "x_indices": np.arange(count, dtype=np.int32) + generation * 100,
        "diagnostics": {"thin_trace": np.arange(max(2, count // 16), dtype=np.float32) + generation},
    }
    return {
        "_id": identifier,
        "origin": "storage-recovery-fixture",
        "mode": "gradation",
        "analysis_id": "recovery-protocol-v1",
        "recsys_type": "OpinionM9",
        "decay": 0.05,
        "rewiring": 0.05,
        "repost_rate": 0.1,
        "complete": True,
        "stale": False,
        "state_epoch": 0,
        "metadata": {"generation": generation, "arrays": array_manifest(payload)},
        "payload": payload,
    }


def operations(payload_bytes: int) -> list[dict]:
    """Every fault has two immutable ACKed anchors plus a mutable document."""
    return [
        {"kind": "put", "key": "anchor-a", "document": make_document("anchor-a", 0, payload_bytes)},
        {"kind": "put", "key": "anchor-b", "document": make_document("anchor-b", 1, payload_bytes)},
        {"kind": "put", "key": "mutable", "document": make_document("mutable", 0, payload_bytes)},
        {"kind": "put", "key": "mutable", "document": make_document("mutable", 1, payload_bytes)},
        {"kind": "update_state", "key": "mutable", "state": {"complete": False, "stale": True, "state_epoch": 1}},
        {"kind": "put", "key": "unconfirmed", "document": make_document("unconfirmed", 2, payload_bytes)},
    ]


def _apply(expected: dict[str, dict], operation: dict) -> None:
    key = operation["key"]
    if operation["kind"] == "put":
        expected[key] = deepcopy(operation["document"])
    else:
        expected[key] = {**expected[key], **operation["state"]}


def validate_document(document: dict) -> None:
    payload = document.get("payload")
    if not isinstance(payload, dict) or not array_manifest(payload):
        raise RecoveryFailure(f"missing or undecoded array payload: {document.get('_id')}")
    if document.get("metadata", {}).get("arrays") != array_manifest(payload):
        raise RecoveryFailure(f"array/metadata generation mismatch: {document.get('_id')}")
    if type(document.get("complete")) is not bool or type(document.get("stale")) is not bool:
        raise RecoveryFailure(f"non-boolean completion state: {document.get('_id')}")
    if document["complete"] == document["stale"]:
        raise RecoveryFailure(f"partial complete/stale update: {document.get('_id')}")


def verify_documents(actual: list[dict], plan: list[dict], acknowledged: list[int], uncertain: int | None) -> dict:
    """Accept only the ACKed prefix or one subsequent, fully committed operation.

    The uncertain operation may replace an existing key, so an unacknowledged
    *complete new version* is permitted; an arbitrary mixture of versions is not.
    """
    if acknowledged != list(range(len(acknowledged))):
        raise RecoveryFailure("ACKs do not form an ordered operation prefix")
    if uncertain is not None and uncertain != len(acknowledged):
        raise RecoveryFailure("the unacknowledged operation is not next after the ACK prefix")
    confirmed = {}
    for index in acknowledged:
        _apply(confirmed, plan[index])
    candidates = [confirmed]
    if uncertain is not None:
        pending = deepcopy(confirmed)
        _apply(pending, plan[uncertain])
        candidates.append(pending)
    observed = {}
    for document in actual:
        validate_document(document)
        key = document.get("_id")
        if key in observed:
            raise RecoveryFailure(f"duplicate recovered key: {key}")
        observed[key] = document_hash(document)
    valid = [{key: document_hash(doc) for key, doc in candidate.items()} for candidate in candidates]
    if observed not in valid:
        missing = sorted(set(confirmed) - set(observed))
        raise RecoveryFailure(
            f"recovered data is neither the ACKed prefix nor the next whole operation; missing={missing}"
        )
    return {
        "documents": len(observed),
        "acknowledged_keys": len(confirmed),
        "unacknowledged_operation_visible": uncertain is not None and observed != valid[0],
        "document_hashes": observed,
    }


def _require_durability(pragmas: dict) -> None:
    if str(pragmas.get("journal_mode", "")).lower() != "wal":
        raise RecoveryFailure(f"recovery requires journal_mode=WAL: {pragmas}")
    if str(pragmas.get("synchronous", "")).upper() not in ("2", "FULL"):
        raise RecoveryFailure(f"recovery requires synchronous=FULL: {pragmas}")


def _writer(backend_name: str, database: str, plan: list[dict], target: int, phase: str, pipe) -> None:
    from .backends import open_backend

    try:
        with open_backend(backend_name, Path(database)) as backend:
            pragmas = backend.pragmas()
            _require_durability(pragmas)
            pipe.send({"event": "ready", "pid": os.getpid(), "pragmas": pragmas})
            expected = {}
            for index, operation in enumerate(plan):
                pipe.send({"event": "write_start", "index": index})
                if pipe.recv() != "proceed":
                    raise RecoveryFailure("writer received an invalid scheduling command")

                def before_commit():
                    pipe.send({"event": "before_commit", "index": index})
                    # This gate stays inside the adapter's uncommitted transaction.
                    if pipe.recv() != "commit":
                        raise RecoveryFailure("writer received an invalid commit command")

                backend.recovery_before_commit = before_commit if phase == "before_commit" and index == target else None
                if operation["kind"] == "put":
                    backend.put(operation["document"])
                else:
                    backend.update_state(operation["key"], operation["state"])
                backend.recovery_before_commit = None
                _apply(expected, operation)
                pipe.send(
                    {
                        "event": "ack",
                        "index": index,
                        "key": operation["key"],
                        "document_hash": document_hash(expected[operation["key"]]),
                    }
                )
                if pipe.recv() != "continue":
                    raise RecoveryFailure("writer received an invalid ACK continuation")
            pipe.send({"event": "finished"})
    except BaseException:
        try:
            pipe.send({"event": "error", "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()


def _receive(pipe, process, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipe.poll(min(0.1, max(0, deadline - time.monotonic()))):
            try:
                message = pipe.recv()
            except EOFError as error:
                raise RecoveryFailure(
                    f"writer exited without the expected event (exitcode={process.exitcode})"
                ) from error
            if message.get("event") == "error":
                raise RecoveryFailure(message["traceback"])
            return message
        if not process.is_alive():
            raise RecoveryFailure(f"writer exited before the selected kill point (exitcode={process.exitcode})")
    raise RecoveryFailure("writer did not reach the selected recovery phase before timeout")


def _kill_writer(process) -> None:
    if process.pid is None or process.pid <= 0 or process.pid == os.getpid():
        raise RecoveryFailure("refusing to signal anything except the writer child")
    if not process.is_alive():
        raise RecoveryFailure("writer exited before SIGKILL")
    os.kill(process.pid, signal.SIGKILL)
    process.join(timeout=10)
    if process.is_alive() or process.exitcode != -signal.SIGKILL:
        raise RecoveryFailure(f"writer was not terminated by SIGKILL: {process.exitcode}")


def run_recovery_round(
    backend_name: str, directory, *, phase: str, target: int, payload_bytes: int = 262144, timeout: float = 30
) -> dict:
    """Run one bounded fault case in a fresh, isolated directory."""
    from .backends import open_backend

    if backend_name not in BACKENDS or phase not in ("after_ack", "before_commit"):
        raise ValueError("unknown backend or fault phase")
    if target not in (2, 3, 4, 5) or timeout <= 0:
        raise ValueError("target must be 2..5 and timeout positive")
    if not hasattr(signal, "SIGKILL"):
        raise RuntimeError("this protocol requires POSIX SIGKILL")
    source_files = (Path(__file__), Path(__file__).with_name("backends.py"))
    if backend_name == "sqlite_optimized":
        source_files += (Path(__file__).with_name("sqlite_optimized.py"),)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    database = directory / "database"
    plan = operations(payload_bytes)
    context = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = context.Pipe(duplex=True)
    process = context.Process(target=_writer, args=(backend_name, str(database), plan, target, phase, child_pipe))
    acknowledged = []
    known = {}
    pragmas = None
    process.start()
    child_pipe.close()
    try:
        with (directory / "ack.jsonl").open("x", encoding="utf-8") as log:
            while True:
                event = _receive(parent_pipe, process, timeout)
                kind = event.get("event")
                if kind == "ready":
                    if event["pid"] != process.pid:
                        raise RecoveryFailure("writer PID does not match its process handle")
                    pragmas = event["pragmas"]
                    _require_durability(pragmas)
                elif kind == "write_start":
                    if event["index"] != len(acknowledged):
                        raise RecoveryFailure("writer attempted an operation outside the ACK sequence")
                    parent_pipe.send("proceed")
                elif kind == "before_commit":
                    if phase != "before_commit" or event["index"] != target or target != len(acknowledged):
                        raise RecoveryFailure("unexpected pre-commit hook")
                    _kill_writer(process)
                    break
                elif kind == "ack":
                    index = event["index"]
                    if index != len(acknowledged):
                        raise RecoveryFailure("duplicate or out-of-order ACK")
                    _apply(known, plan[index])
                    if event["key"] != plan[index]["key"] or event["document_hash"] != document_hash(
                        known[event["key"]]
                    ):
                        raise RecoveryFailure("writer ACK does not match the deterministic operation")
                    log.write(json.dumps(event, sort_keys=True) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    acknowledged.append(index)
                    if phase == "after_ack" and index == target:
                        _kill_writer(process)
                        break
                    parent_pipe.send("continue")
                else:
                    raise RecoveryFailure(f"unexpected writer event: {event}")
    finally:
        if process.is_alive():
            _kill_writer(process)
        parent_pipe.close()
        process.join(timeout=10)
    # Read the parent's actual durable journal, not only the in-memory ACK list.
    ack_path = directory / "ack.jsonl"
    ack_events = [json.loads(line) for line in ack_path.read_text().splitlines()]
    if [event["index"] for event in ack_events] != acknowledged:
        raise RecoveryFailure("ACK journal differs from the coordinator's confirmed prefix")
    with open_backend(backend_name, database, readonly=True) as backend:
        actual = backend.query({}, projection=False)
        integrity = backend.integrity_check()
        if integrity != ["ok"]:
            raise RecoveryFailure(f"SQLite integrity_check failed: {integrity}")
        if backend.count() != len(actual):
            raise RecoveryFailure("recovered count differs from recovered documents")
        for document in actual:
            fetched = backend.get(document["_id"], include_payload=True)
            if fetched is None or document_hash(fetched) != document_hash(document):
                raise RecoveryFailure("get and query disagree after independent reopen")
    checked = verify_documents(actual, plan, acknowledged, target if phase == "before_commit" else None)
    if source_hashes != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}:
        raise RecoveryFailure("backend or recovery source changed during the fault round")
    result = {
        "backend": backend_name,
        "phase": phase,
        "target_operation": target,
        "target_kind": plan[target]["kind"],
        "fault": "writer_process_SIGKILL",
        "power_loss_test": False,
        "coordinator_pid": os.getpid(),
        "writer_pid": process.pid,
        "writer_exitcode": process.exitcode,
        "acknowledged_operations": len(acknowledged),
        "ack_log_sha256": hashlib.sha256(ack_path.read_bytes()).hexdigest(),
        "implementation_sha256": source_hashes,
        "pragmas": pragmas,
        "integrity_check": integrity,
        "passed": True,
        **checked,
    }
    (directory / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def run_recovery(
    backend_name: str, directory, *, rounds: int = 6, payload_bytes: int = 262144, timeout: float = 30
) -> dict:
    """Run at least five independent fault rounds; preserve each round's evidence."""
    if backend_name not in BACKENDS:
        raise ValueError(f"unknown backend: {backend_name}")
    if type(rounds) is not int or rounds < 5:
        raise ValueError("the recovery protocol requires at least five independent rounds")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("recovery output must be a fresh empty directory")
    results = []
    for number in range(rounds):
        phase, target = SCHEDULE[number % len(SCHEDULE)]
        try:
            result = run_recovery_round(
                backend_name,
                directory / f"round-{number:03}",
                phase=phase,
                target=target,
                payload_bytes=payload_bytes,
                timeout=timeout,
            )
        except Exception as error:
            result = {
                "backend": backend_name,
                "phase": phase,
                "target_operation": target,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append({"round": number, **result})
    report = {
        "schema": 1,
        "backend": backend_name,
        "fault": "writer_process_SIGKILL",
        "power_loss_test": False,
        "planned_rounds": rounds,
        "completed_rounds": len(results),
        "passed_rounds": sum(r["passed"] for r in results),
        "all_passed": all(r["passed"] for r in results),
        "results": results,
    }
    (directory / "recovery.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument(
        "--directory", type=Path, required=True, help="Fresh isolated recovery output, never a production database."
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--payload-bytes", type=int, default=262144)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    report = run_recovery(
        args.backend, args.directory, rounds=args.rounds, payload_bytes=args.payload_bytes, timeout=args.timeout
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
