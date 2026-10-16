"""Scientific NPZ and micro/pair adapters for the common task lifecycle."""

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from common.experiments.execution import TaskAdapter
from common.experiments.plans import freeze_manifest
from common.io.json import fingerprint, sha
from common.io.records import read_record, record_path, write_record
from common.runtime.failures import ProtocolError, TaskError
from common.runtime.process import run_isolated
from common.types import RuntimeBinding
from theory.parameters import Parameters
from theory.runtime import measure
from theory.runtime.archives import save_result
from theory.runtime.completed import cached
from theory.runtime.population import binary_result, pair_identity, population_request
from theory.types import MeasureTask, MicroTask, PairRecord, PairTask, PairTrace, SharedPairResult


def kinetic_batch(binary, requests, workers):
    from smp_meso_bindings import run_kinetic_batch_parallel

    return run_kinetic_batch_parallel(binary, requests, workers, check=False)


class MeasureAdapter(TaskAdapter):
    def __init__(self, directory: Path, binary, store=None, *, binding_batch=False):
        self.directory, self.binary, self.store = Path(directory), binary, store
        binding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(measure.__file__))}

        def operation():
            return freeze_manifest(self.directory / "binding.json", binding)

        store.read(operation) if store else operation()
        super().__init__(
            lambda t: t["id"],
            self.is_cached,
            self.compute_result,
            self.save_result,
            batch=self.run_binding_batch if binding_batch else None,
        )

    def path(self, task):
        return self.directory / "raw" / (task["id"] + ".npz")

    def is_cached(self, task):
        path = self.path(task)
        return path.exists() and cached(path, task, "measure")

    def compute_result(self, task: MeasureTask):
        result = measure.run_measure(
            self.binary, task["id"], Parameters(**task["parameters"]), **task.get("request_options", {})
        )
        diagnostics = result["diagnostics"]
        if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
            raise TaskError(f"mass/degree conservation failed: {task['id']}")
        return result

    def run_binding_batch(self, tasks, workers, complete, failed):
        # Bound response memory to a worker-sized chunk. This API and request
        # schema are known only to this model adapter, never to common.experiments.
        for start in range(0, len(tasks), workers):
            group = tasks[start : start + workers]
            requests = [
                measure.measure_request(t["id"], Parameters(**t["parameters"]), **t.get("request_options", {}))
                for t in group
            ]
            responses = run_isolated(kinetic_batch, self.binary, requests, min(workers, len(group)))
            expected = {t["id"]: t for t in group}
            seen = set()
            for response in responses:
                identifier = response.get("request_id")
                if identifier not in expected or identifier in seen:
                    raise ProtocolError("invalid or duplicate kinetic batch response identity")
                seen.add(identifier)
                task = expected[identifier]
                result = response.get("result")
                if response.get("error") or not isinstance(result, dict):
                    failed(task, TaskError(f"{identifier}: invalid kinetic response: {response.get('error')}"))
                    continue
                diagnostics = result["diagnostics"]
                if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
                    failed(task, TaskError(f"mass/degree conservation failed: {identifier}"))
                    continue
                complete(task, result)
            for identifier in expected.keys() - seen:
                failed(expected[identifier], TaskError(f"missing kinetic response: {identifier}"))

    def save_result(self, task, result):
        def writer(staged):
            return save_result(staged, task, result)

        self.store.commit(self.path(task), writer) if self.store else writer(self.path(task))


class PopulationAdapter(TaskAdapter):
    def __init__(self, directory: Path, binary, store=None, *, force_pairs=False):
        self.directory, self.binary, self.store = Path(directory), binary, store
        self.force_pairs = force_pairs
        self.binding: RuntimeBinding = {
            "binary_sha256": sha(binary),
            "request_schema_sha256": sha(Path(measure.__file__)),
        }
        self.pair_directory = self.directory.parent / "pairs"
        self.locks: dict[str, threading.Lock] = {}
        self.lock_guard = threading.Lock()
        self.pair_summaries: dict[str, tuple[str, str]] = {}
        self.read(lambda: self.directory.mkdir(parents=True, exist_ok=True))
        self.read(lambda: freeze_manifest(self.directory.parent / "binding.json", self.binding))
        super().__init__(lambda t: t["id"], self.is_cached, self.compute_result, self.save_result)

    def read(self, operation):
        return self.store.read(operation) if self.store else operation()

    def commit(self, path: Path, task, result: Mapping[str, Any]):
        def writer(staged):
            return write_record(staged, task, result)

        self.store.commit(path, writer) if self.store else writer(path)

    def check(self, path: Path, task, kind: Literal["micro", "pair"]):
        return self.read(lambda: path.exists() and cached(path, task, kind))

    def is_cached(self, task):
        path = self.read(lambda: record_path(self.directory, task["id"]))
        return self.check(path, task, "micro")

    def compute_result(self, task: MicroTask):
        payload = population_request(task)
        identity = pair_identity(task, self.binding)
        key = fingerprint(identity)
        pair_task: PairTask = {"id": key, **identity}
        pair_path = self.pair_directory / (key + ".msgpack.lz4")
        with self.lock_guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self.pair_summaries:
                if self.force_pairs or not self.check(pair_path, pair_task, "pair"):
                    prediction = binary_result(
                        self.binary, {**payload, "id": key, "kind": "pair_only"}, self.store, self.directory
                    )
                    pair_result: SharedPairResult = {
                        "initial_hash": prediction["initial_hash"],
                        "projection_l1": prediction["projection_l1"],
                        "pair": cast(PairTrace, prediction["pair"]),
                    }
                    diagnostics = pair_result["pair"]["diagnostics"]
                    if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
                        raise TaskError(f"{key}: pair conservation failed")
                    self.commit(pair_path, pair_task, pair_result)
                pair = cast(PairRecord, self.read(lambda: read_record(pair_path)))
                self.pair_summaries[key] = (pair["result"]["initial_hash"], self.read(lambda: sha(pair_path)))
            initial_hash, pair_sha = self.pair_summaries[key]
        result = binary_result(self.binary, {**payload, "kind": "micro_only"}, self.store, self.directory)
        if result["initial_hash"] != initial_hash:
            raise ProtocolError(f"{task['id']}: shared pair initial state changed")
        result.pop("pair", None)
        result["pair_ref"] = {"key": key, "sha256": pair_sha}
        return result

    def save_result(self, task, result):
        self.commit(self.directory / (task["id"] + ".msgpack.lz4"), task, result)
