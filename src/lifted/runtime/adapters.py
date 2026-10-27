"""TaskAdapter implementation over shared initialization and public continuation APIs."""

import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

from common.experiments.execution import TaskAdapter
from common.io.json import sha
from common.io.records import read_transport, write_record
from common.runtime.failures import ProtocolError
from common.runtime.process import run_binary
from lifted.runtime import records
from lifted.runtime.protocol import request, stream_seed
from theory.parameters import Parameters
from theory.runtime.population import binary_result, population_request
from theory.types import MicroTask


class BudgetExhausted(RuntimeError):
    """Operational deadline; never a physical outcome label."""


class ContinuationAdapter(TaskAdapter):
    def __init__(self, root, tasks, binaries, binding, store, *, deadline=None):
        self.root, self.tasks = Path(root), tasks
        self.binaries, self.binding, self.store = binaries, binding, store
        self.deadline = deadline
        super().__init__(lambda t: t["id"], self.is_cached, self.compute_result, self.save_result)

    def is_cached(self, task):
        if not records.record_path(self.root, task["id"]).exists():
            return False
        records.read(self.root, task, self.binding)
        return True

    def call(self, payload, directory):
        remaining = None if self.deadline is None else self.deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise BudgetExhausted("wall budget exhausted; task remains pending")
        path = directory / "response.msgpack"
        run_binary(
            self.binaries["lifted"],
            json.dumps(payload) + "\n",
            path,
            timeout=remaining,
            max_bytes=self.store.max_record_bytes,
        )
        return read_transport(path, payload["id"])

    def initial(self, task, directory):
        p = replace(Parameters(**task["parameters"]), horizon=0.02, record_tau=0.02)
        # Reuse the exact theory initialization family and stream policy.
        micro = cast(
            MicroTask,
            dict(
                id=task["id"],
                kind="micro",
                parameters=vars(p),
                request_options={},
                roles=[],
                initial=task["initial"],
                branch=task["mother"],
                initial_seed=stream_seed(task["seed"], "initial", task["initial"]),
                process_seed=stream_seed(task["seed"], "mother", task["initial"], task["mother"]),
                selection="weighted",
            ),
        )
        payload = population_request(micro)
        payload["kind"], payload["checkpoint_dir"] = "micro_only", str(directory / "initial")
        payload["snapshot_steps"] = [0]
        binary_result(self.binaries["initial"], payload, self.store, directory)
        path = directory / "initial" / "checkpoint-000000000.msgpack.lz4"
        source = dict(checkpoint=str(path), checkpoint_sha256=sha(path))
        operation = {**task, "kind": "micro_atlas", "horizon_steps": 0, "observe_steps": [0]}
        payload = request(operation, source, directory / "states", source_kind="micro", mode="resume")
        if task["initial_law"] == "bridge":
            payload["bridge_agents"] = task["bridge_agents"]
        return self.call(payload, directory)

    def compute_result(self, task):
        started = time.monotonic()
        directory = Path(tempfile.mkdtemp(prefix="lifted-task-", dir=self.store.spool))
        try:
            dependency = None
            if task["kind"] == "initial":
                result = self.initial(task, directory)
            else:
                parent = self.tasks[task["source_task"]]
                source, dependency = records.source_reference(self.root, task, parent)
                source_kind = "meso" if parent["kind"].startswith("meso_") else "micro"
                mode = (
                    "resume"
                    if task["kind"].endswith("extend") or task["kind"] in ("micro_atlas", "micro_tail")
                    else "branch"
                )
                payload = request(task, source, directory / "states", source_kind=source_kind, mode=mode)
                if task["kind"] == "micro_tail":
                    payload["kind"] = "micro_extend"
                result = self.call(payload, directory)
                if task["kind"] == "micro_atlas" and parent["kind"] == "micro_atlas":
                    original = records.read(self.root, parent)
                    known = {row["step"]: row["physical_hash"] for row in original["states"]}
                    if any(
                        row["step"] in known and row["physical_hash"] != known[row["step"]] for row in result["states"]
                    ):
                        raise ProtocolError("replayed mother path differs at an original checkpoint")
            result.update(binding=self.binding, dependency_sha256=dependency, wall_seconds=time.monotonic() - started)
            result["_temporary"] = str(directory)
            for path in result["paths"]:
                if path.get("delta") is not None:
                    path["delta"] = np.asarray(path["delta"], dtype=np.float64)
            return result
        except BaseException:
            shutil.rmtree(directory)
            raise

    def save_result(self, task, result):
        directory = Path(result.pop("_temporary"))
        try:
            result["files"] = {}
            for row in [*result["states"], *result["paths"]]:
                if not row.get("checkpoint"):
                    continue
                original = Path(row["checkpoint"])
                relative = Path("states") / task["id"] / original.name
                if sha(original) != row["checkpoint_sha256"]:
                    raise ProtocolError("checkpoint changed before commit")
                self.store.commit(self.root / relative, lambda staged, source=original: shutil.copyfile(source, staged))
                row["checkpoint"] = str(relative)
                result["files"][str(relative)] = row["checkpoint_sha256"]
            self.store.commit(
                records.record_path(self.root, task["id"]), lambda staged: write_record(staged, task, result)
            )
        finally:
            shutil.rmtree(directory)
