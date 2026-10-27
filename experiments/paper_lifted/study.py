"""Study stages delegate execution, persistence and progress to the shared lifecycle."""

import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from common.experiments.execution import TaskAdapter, execute_ready_tasks, execute_tasks
from common.experiments.lifecycle import StageJournals, TaskJournal, Workflow
from common.io.json import atomic_json
from common.runtime.failures import TaskDeferred
from lifted.runtime.adapters import BudgetExhausted, ContinuationAdapter
from lifted.runtime.records import read


class LiftedStudy:
    def __init__(
        self, root, plan, binaries, binding, store, *, workers, deadline=None, schedule="stages", progress=None
    ):
        self.root, self.plan, self.workers = Path(root), plan, workers
        self.deadline = deadline
        self.schedule, self.progress = schedule, progress
        self.tasks = {t["id"]: t for t in plan["tasks"]}
        self.adapter = ContinuationAdapter(root, self.tasks, binaries, binding, store, deadline=deadline)

    def compute(self, task):
        try:
            return self.adapter.compute(task)
        except (BudgetExhausted, subprocess.TimeoutExpired) as error:
            if isinstance(error, BudgetExhausted) or (self.deadline is not None and time.monotonic() >= self.deadline):
                raise TaskDeferred("wall budget exhausted; task remains pending") from error
            raise

    def save(self, task, result):
        self.adapter.save(task, result)
        if self.progress:
            try:
                self.progress.observe(task, result)
            except Exception as error:
                print(f"[overall] timing observation unavailable: {error}", file=sys.stderr, flush=True)

    def stage(self, stage):
        directory = self.root / "stages" / stage["name"]
        journal = TaskJournal(directory)
        selected = [self.tasks[i] for i in stage["task_ids"]]
        for start in range(0, len(selected), self.workers):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            execute_tasks(
                selected[start : start + self.workers],
                TaskAdapter(lambda t: t["id"], self.adapter.cached, self.compute, self.save),
                workers=self.workers,
                store=self.adapter.store,
                journal=journal,
                progress=self.progress or True,
            )
        complete = [t["id"] for t in selected if self.adapter.is_cached(t)]
        # A fresh journal contains only this attempt; coverage includes earlier committed work.
        from common.experiments.plans import coverage

        result = {**coverage(selected, complete), "state": "complete" if len(complete) == len(selected) else "partial"}
        atomic_json(directory / "coverage.json", result)
        return result

    def run(self):
        if self.schedule == "balanced":
            return Workflow(
                {"balanced": lambda: self.balanced(self.plan["stages"])},
                self.root / "status.json",
                profile=self.plan["profile"],
                workers=self.workers,
            ).dispatch("balanced")
        workflow = Workflow(
            {s["name"]: partial(self.stage, s) for s in self.plan["stages"]},
            self.root / "status.json",
            profile=self.plan["profile"],
            workers=self.workers,
        )
        for stage in self.plan["stages"]:
            result = workflow.dispatch(stage["name"])
            if result["missing"]:
                return result
        return {"missing": []}

    def balanced(self, stages, *, prefix=""):
        from .scheduling import fair_key

        selected = [self.tasks[i] for stage in stages for i in stage["task_ids"]]
        complete = {task["id"] for task in self.tasks.values() if self.adapter.is_cached(task)}
        if self.progress:
            self.progress.verified(complete)
        journals = StageJournals(self.root / "stages", stages, prefix=prefix)
        adapter = TaskAdapter(lambda t: t["id"], self.adapter.cached, self.compute, self.save)
        completed = execute_ready_tasks(
            selected,
            adapter,
            completed=complete,
            key=fair_key,
            workers=self.workers,
            deadline=self.deadline,
            store=self.adapter.store,
            journal=journals,
            progress=self.progress or False,
            on_finish=journals.publish,
        )
        return {"missing": [t["id"] for t in selected if t["id"] not in completed]}

    def followup(self):
        """Delegate scientific selection to evidence, then reuse the selected execution policy."""
        from .evidence.refinement import freeze_followup

        def read_result(task):
            return self.adapter.is_cached(task) and self.adapter.store.read(lambda: read(self.root, task))

        followup = freeze_followup(self.root, self.plan, self.tasks, read_result=read_result)
        if followup is None:
            return {"missing": []}
        self.tasks.update({t["id"]: t for t in followup["tasks"]})
        self.adapter.tasks = self.tasks
        if self.progress:
            self.progress.add_tasks(followup["tasks"])
        if self.schedule == "balanced":
            return self.balanced(followup["stages"], prefix="followup_")
        for stage in followup["stages"]:
            result = self.stage({**stage, "name": "followup_" + stage["name"]})
            if result["missing"]:
                return result
        return {"missing": []}
