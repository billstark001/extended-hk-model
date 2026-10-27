"""Pure continuation-work forecasting from timings and supplied state metadata."""

import statistics
from collections import defaultdict, deque

from .costs import task_cost


class ContinuationCosts:
    """Predictions use pilot costs, observed native rates and original mother futures.

    Native record metadata is read for operational estimation only. Progress never
    scans/checksums all checkpoints and does not replace scientific validation.
    """

    def __init__(self, tasks, *, metadata, measurements=(), baseline_seconds_by_kind=None):
        self.tasks, self.reader = tasks, metadata
        self.metadata, self.fallback = {}, {}
        self.baseline = baseline_seconds_by_kind or {}
        self.rates, self.branch_steps = defaultdict(lambda: deque(maxlen=128)), defaultdict(lambda: deque(maxlen=64))
        self.observed, self.observed_wall = set(), set()
        self.wall_by_source, self.wall_by_stratum = (
            defaultdict(lambda: deque(maxlen=64)),
            defaultdict(lambda: deque(maxlen=128)),
        )
        self.measurements = list(measurements)
        self.truth = {t["source_task"]: t["id"] for t in tasks.values() if t["kind"] == "micro_tail"}

    def read(self, identifier):
        if identifier in self.metadata:
            return self.metadata[identifier]
        result = self.reader(identifier)
        if result is not None:
            if len(self.metadata) > 1024:
                self.metadata.clear()
            self.metadata[identifier] = result
        return result

    def observe(self, task, result):
        identifier = task["id"]
        if identifier in self.observed:
            return
        self.observed.add(identifier)
        if task["kind"] in ("initial", "micro_atlas", "micro_tail", "meso_atlas"):
            self.metadata[identifier] = result
        if task["kind"] in ("micro_probe", "micro_extend") and result["executed_steps"]:
            self.rates[task["parameters"]["population"]].append(result["elapsed_seconds"] / result["executed_steps"])
        if task["kind"] == "micro_probe":
            self.branch_steps[(task["source_task"], task["source_step"])].append(
                result["executed_steps"] / len(task["replicates"])
            )

    @staticmethod
    def stratum(task):
        p = task["parameters"]
        return (
            task["kind"],
            p["population"],
            p["bins"],
            p["ratio"],
            task["variant"],
            task["layer"],
            task["source_step"],
            task["horizon_steps"],
        )

    def observe_wall(self, task, seconds):
        if task["id"] in self.observed_wall or seconds <= 0:
            return
        self.observed_wall.add(task["id"])
        key = self.stratum(task)
        per_path = seconds / len(task["replicates"])
        self.wall_by_source[(key, task["source_task"])].append(per_path)
        self.wall_by_stratum[key].append(per_path)

    def bootstrap(self, index):
        # Both early and late representatives; do not let cached events overwrite measured durations.
        groups = defaultdict(list)
        for identifier in index.complete & self.tasks.keys():
            t = self.tasks[identifier]
            self.observe_wall(t, index.durations.get(identifier, 0))
            if t["kind"] == "micro_probe" and index.durations.get(identifier, 0) > 1:
                groups[(t["parameters"]["population"], t["parameters"]["ratio"])].append(identifier)
        for group in groups.values():
            group.sort(key=lambda identifier: index.finished.get(identifier, 0))
            sample = {group[round(i * (len(group) - 1) / 23)] for i in range(24)}
            for identifier in sample:
                self.observe(self.tasks[identifier], self.read(identifier))

    def predict(self, task):
        p = task["parameters"]
        if task["kind"] == "micro_probe":
            parent = self.read(task["source_task"])
            state = next((s for s in parent["states"] if s["step"] == task["source_step"]), None) if parent else None
            if state and state["category"] != "unresolved":
                return 0.1
            matched = self.wall_by_source.get((self.stratum(task), task["source_task"]))
            if matched:
                return statistics.median(matched) * len(task["replicates"])
            hit = None
            if parent and parent["paths"]:
                hit = parent["paths"][0].get("first_hit_step")
            if hit is None and task["source_task"] in self.truth:
                future = self.read(self.truth[task["source_task"]])
                if future and future["paths"][0]["category"] != "unresolved":
                    hit = future["paths"][0]["step"]
            observed = self.branch_steps.get((task["source_task"], task["source_step"]))
            if self.rates[p["population"]] and (hit is not None or observed):
                steps = statistics.mean(observed) if observed else max(1, hit - task["source_step"])
                return 0.02 + min(steps, task["horizon_steps"]) * len(task["replicates"]) * statistics.median(
                    self.rates[p["population"]]
                )
        observed = self.wall_by_source.get((self.stratum(task), task["source_task"])) or self.wall_by_stratum.get(
            self.stratum(task)
        )
        if observed:
            return statistics.median(observed) * len(task["replicates"])
        key = (
            task["kind"],
            p["ratio"],
            p["population"],
            p["bins"],
            task["variant"],
            task["layer"],
            task["source_step"],
            task["horizon_steps"],
            len(task["replicates"]),
        )
        if key not in self.fallback:
            try:
                self.fallback[key] = task_cost(task, self.measurements)[0]
            except ValueError:
                by_kind = self.baseline
                n = sum(t["kind"] == task["kind"] for t in self.tasks.values())
                self.fallback[key] = by_kind.get(task["kind"], 0) / max(1, n) or None
        return self.fallback[key]
