"""Single-simulation SMP protocol adapter; scheduling belongs to common.experiments."""

import json
import sys
import threading
from copy import deepcopy
from dataclasses import fields
from ehk.domain import Task, parse_metadata
from pathlib import Path

from common.experiments.execution import TaskAdapter
from common.runtime.process import cancellation_event, run_isolated


class SimulationAdapter(TaskAdapter):
    def __init__(self, binary, workspace):
        self.binary, self.workspace = Path(binary), Path(workspace)
        self.print_lock, self.active_lock = threading.Lock(), threading.Lock()
        self.active_processes: set = set()
        super().__init__(
            lambda t: t.id,
            self.is_cached,
            self.run_one,
            self.persisted,
            self.cancel_active,
        )

    def is_cached(self, task):
        from smp_bindings.simulation import is_simulation_finished

        self.validate_existing(task)
        return is_simulation_finished(str(self.workspace), task.metadata)

    def validate_existing(self, task):
        """Prevent a seeded recalculation from adopting unrelated legacy output."""
        directory = self.workspace / task.metadata.UniqueName
        path = directory / "metadata.json"
        if not path.exists():
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"existing output has no reproducible metadata: {directory}; use a new workspace")
            return
        saved = parse_metadata(json.loads(path.read_text()))
        changed = []
        for item in fields(task.metadata):
            if item.name.startswith("_"):
                continue
            requested = getattr(task.metadata, item.name)
            if requested is None:
                continue
            actual = getattr(saved, item.name)
            if item.name == "ViewPostsDetails" and requested is False and actual is None:
                continue  # The native wire declaration omits this false flag.
            if actual != requested:
                changed.append(item.name)
        if changed:
            raise ValueError(f"existing simulation differs in {', '.join(changed)}: {directory}; use a new workspace")

    def run_one(self, task):
        from smp_bindings.simulation import run_simulation

        self.validate_existing(task)
        return run_simulation(
            str(self.binary),
            str(self.workspace),
            deepcopy(task.metadata),
            sys.stdout.isatty(),
            self.print_lock,
            stop_event=cancellation_event(),
            active_procs=self.active_processes,
            active_procs_lock=self.active_lock,
        )

    def persisted(self, task, result):
        if result != task.id or not self.is_cached(task):
            raise ValueError(f"simulation did not commit a finished record: {task.id}")

    def cancel_active(self):
        from smp_bindings.simulation import terminate_process

        with self.active_lock:
            active = list(self.active_processes)
        for process in active:
            terminate_process(process, process_group=True)
