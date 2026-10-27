"""Study stages delegate execution, persistence and progress to the shared lifecycle."""

import time
from functools import partial
from pathlib import Path

from common.experiments.execution import execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow
from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, sha
from common.runtime.failures import ProtocolError
from lifted.runtime.adapters import ContinuationAdapter
from lifted.runtime.records import read


class LiftedStudy:
    def __init__(self, root, plan, binaries, binding, store, *, workers, deadline=None):
        self.root, self.plan, self.workers = Path(root), plan, workers
        self.deadline = deadline
        self.tasks = {t["id"]: t for t in plan["tasks"]}
        self.adapter = ContinuationAdapter(root, self.tasks, binaries, binding, store, deadline=deadline)

    def stage(self, stage):
        directory = self.root / "stages" / stage["name"]
        journal = TaskJournal(directory)
        selected = [self.tasks[i] for i in stage["task_ids"]]
        for start in range(0, len(selected), self.workers):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            execute_tasks(
                selected[start : start + self.workers],
                self.adapter,
                workers=self.workers,
                store=self.adapter.store,
                journal=journal,
            )
        complete = [t["id"] for t in selected if self.adapter.is_cached(t)]
        # A fresh journal contains only this attempt; coverage includes earlier committed work.
        from common.experiments.plans import coverage

        result = {**coverage(selected, complete), "state": "complete" if len(complete) == len(selected) else "partial"}
        atomic_json(directory / "coverage.json", result)
        return result

    def run(self):
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

    def followup(self):
        """Freeze independent validation and same-stream tail tasks after the core run.

        Parent records and their hashes authenticate data-dependent selection.
        The root plan remains immutable; a resumed follow-up reads its own plan.
        """
        from theory.parameters import Parameters

        from .evidence.analyze import probability_groups
        from .evidence.design import Design, chunked

        path = self.root / "followup" / "plan.json"
        if path.exists():
            followup = read_plan(path)
            for relative, digest in followup["inputs"].items():
                if sha(self.root / relative) != digest:
                    raise ProtocolError("follow-up selection inputs changed")
        else:
            d = Design("formal")
            for task in list(self.tasks.values()):
                if task["kind"] not in ("micro_probe", "meso_probe", "micro_tail"):
                    continue
                if task["variant"] != "stochastic":
                    continue
                result = self.adapter.is_cached(task) and self.adapter.store.read(lambda t=task: read(self.root, t))
                if not result:
                    continue
                for row in result["paths"]:
                    if row["category"] != "unresolved":
                        continue
                    p = Parameters(**task["parameters"])
                    d.add(
                        "tails",
                        task["kind"].replace("probe", "extend").replace("tail", "extend"),
                        p,
                        parent=task["id"],
                        initial=task["initial"],
                        mother=task["mother"],
                        at=task["source_step"],
                        start=row["replicate"],
                        paths=1,
                        steps=24000,
                        variant=task["variant"],
                        layer=task["layer"],
                        cohort=task["cohort"],
                    )
            for group in probability_groups(self.root, self.plan):
                if group["kind"] != "micro_probe" or group["unresolved"] or not 0.7 <= max(group["estimate"]) <= 0.99:
                    continue
                task = self.tasks[group["representative_task"]]
                if len(d.stages.get("windows", [])) < 2 and task["cohort"] == "population":
                    parent = read(self.root, self.tasks[task["source_task"]])
                    center = task["source_step"]
                    before = max(s["step"] for s in parent["states"] if s["step"] <= max(0, center - 4))
                    points = list(range(max(before, center - 4), center + 5))
                    fine = d.add(
                        "windows",
                        "micro_atlas",
                        Parameters(**task["parameters"]),
                        parent=task["source_task"],
                        at=before,
                        steps=points[-1] - before,
                        observe=[t - before for t in points],
                        initial=task["initial"],
                        mother=task["mother"],
                        cohort="mechanism",
                    )
                    for at in points:
                        chunked(
                            d,
                            "window_probabilities",
                            "micro_probe",
                            Parameters(**task["parameters"]),
                            parent=fine,
                            at=at,
                            initial=task["initial"],
                            mother=task["mother"],
                            paths=128,
                            steps=24000,
                            cohort="mechanism",
                            seed=20261029,
                        )
                        chunked(
                            d,
                            "window_closure",
                            "meso_probe",
                            Parameters(**task["parameters"]),
                            parent=fine,
                            at=at,
                            initial=task["initial"],
                            mother=task["mother"],
                            paths=64,
                            steps=24000,
                            cohort="mechanism",
                            seed=20261029,
                        )
                chunked(
                    d,
                    "validation",
                    "micro_probe",
                    Parameters(**task["parameters"]),
                    parent=task["source_task"],
                    at=task["source_step"],
                    initial=task["initial"],
                    mother=task["mother"],
                    paths=512,
                    start=100000,
                    steps=24000,
                    cohort="mechanism",
                    seed=20261028,
                )
                # At most six focused cases, kept separate from the population design.
                if len(d.stages.get("validation", [])) >= 6 * 64:
                    break
            if not d.tasks:
                return {"missing": []}
            inputs = {str(p.relative_to(self.root)): sha(p) for p in (self.root / "raw").glob("*.msgpack.lz4")}
            followup = freeze_plan(
                path, {**d.freeze(), "inputs": inputs, "parent_plan_fingerprint": self.plan["plan_fingerprint"]}
            )
        self.tasks.update({t["id"]: t for t in followup["tasks"]})
        self.adapter.tasks = self.tasks
        for stage in followup["stages"]:
            result = self.stage({**stage, "name": "followup_" + stage["name"]})
            if result["missing"]:
                return result
        return {"missing": []}
