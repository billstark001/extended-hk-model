"""Matched micro branches with shared deterministic pair traces and binary records."""

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from common.io.json import atomic_json, fingerprint, sha
from common.io.records import read_record, record_path, write_record
from common.runtime.failures import ProtocolError, TaskError
from common.runtime.process import run_tasks
from common.runtime.storage import RunStorage
from common.types import PathInput, RuntimeBinding
from theory.runtime import measure as schema
from theory.runtime.completed import cached
from theory.runtime.population import binary_result, pair_identity, population_request
from theory.types import MicroTask, PairRecord, PairTask, PairTrace, SharedPairResult


def execute(
    tasks: list[MicroTask],
    directory: PathInput,
    binary: PathInput,
    *,
    force: bool = False,
    limit: int | None = None,
    workers: int = 1,
    store: RunStorage | None = None,
) -> None:
    """Scientific task IDs remain unchanged; pair dependencies have separate keys."""
    directory = Path(directory)
    binding: RuntimeBinding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(schema.__file__))}
    binding_path = directory.parent / "binding.json"

    def bind() -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ProtocolError("population binary changed; use a new output directory")
        atomic_json(binding_path, binding)

    store.read(bind) if store else bind()
    pair_directory = directory.parent / "pairs"
    locks: dict[str, threading.Lock] = {}
    lock_guard = threading.Lock()
    pair_summaries: dict[str, tuple[str, str]] = {}

    def commit(path: Path, task: MicroTask | PairTask, result: Mapping[str, Any]) -> None:
        def writer(staged: Path) -> None:
            return write_record(staged, task, result)

        store.commit(path, writer) if store else writer(path)

    def check(path: Path, task: MicroTask | PairTask, kind: Literal["micro", "pair"]) -> bool:
        def operation() -> bool:
            return path.exists() and cached(path, task, kind)

        return store.read(operation) if store else operation()

    def run(index_task: tuple[int, MicroTask]) -> None:
        index, task = index_task
        old_path = (
            store.read(lambda: record_path(directory, task["id"])) if store else record_path(directory, task["id"])
        )
        if not force and check(old_path, task, "micro"):
            print(f"cached {task['id']}", flush=True)
            return
        payload = population_request(task)
        identity = pair_identity(task, binding)
        key = fingerprint(identity)
        pair_task: PairTask = {"id": key, **identity}
        pair_path = pair_directory / (key + ".msgpack.lz4")
        with lock_guard:
            lock = locks.setdefault(key, threading.Lock())
        with lock:
            if key not in pair_summaries:
                if force or not check(pair_path, pair_task, "pair"):
                    print(f"pair {key}", flush=True)
                    prediction = binary_result(binary, {**payload, "id": key, "kind": "pair_only"}, store, directory)
                    pair_result: SharedPairResult = {
                        "initial_hash": prediction["initial_hash"],
                        "projection_l1": prediction["projection_l1"],
                        "pair": cast(PairTrace, prediction["pair"]),
                    }
                    diagnostics = pair_result["pair"]["diagnostics"]
                    if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
                        raise TaskError(f"{key}: pair conservation failed")
                    commit(pair_path, pair_task, pair_result)
                pair = cast(PairRecord, store.read(lambda: read_record(pair_path)) if store else read_record(pair_path))
                pair_sha = store.read(lambda: sha(pair_path)) if store else sha(pair_path)
                pair_summaries[key] = (pair["result"]["initial_hash"], pair_sha)
            initial_hash, pair_sha = pair_summaries[key]
        print(f"run {index + 1}/{len(tasks)} {task['id']}", flush=True)
        result = binary_result(binary, {**payload, "kind": "micro_only"}, store, directory)
        if result["initial_hash"] != initial_hash:
            raise ProtocolError(f"{task['id']}: shared pair initial state changed")
        result.pop("pair", None)
        result["pair_ref"] = {"key": key, "sha256": pair_sha}
        path = directory / (task["id"] + ".msgpack.lz4")
        commit(path, task, result)

    selected = tasks[:limit] if limit else tasks
    run_tasks(run, enumerate(selected), workers, isolate_failures=True, event=store.event if store else None)
