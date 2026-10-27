"""Read-only journal monitoring and live, stratified cost estimates; no model launch."""

import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from common.experiments.plans import read_plan
from common.experiments.progress import JournalIndex, duration, remaining_work
from common.io.json import atomic_json
from lifted.runtime.forecast import ContinuationCosts

from .evidence.costs import load_pilot_measurements, read_cost_metadata
from .policy import control_directory, reference_workers
from .scheduling import family


class OverallProgress:
    def __init__(self, root, plan, *, workers=None, core_deadline=None, interval=15):
        self.root, self.plan = Path(root), plan
        self.tasks = {t["id"]: t for t in plan["tasks"]}
        followup = self.root / "followup/plan.json"
        if followup.exists():
            extra = read_plan(followup)
            if extra["parent_plan_fingerprint"] != plan["plan_fingerprint"]:
                raise ValueError("progress follow-up belongs to a different frozen plan")
            self.tasks.update({t["id"]: t for t in extra["tasks"]})
        self.control = control_directory(root, plan)
        self.index = JournalIndex(self.control / "events.jsonl", deferred_errors=("wall budget exhausted",))
        self.index.refresh()
        raw = self.root / "raw"
        if raw.exists():
            self.index.complete.update(
                p.name.removesuffix(".msgpack.lz4")
                for p in raw.iterdir()
                if p.name.endswith(".msgpack.lz4") and not p.name.startswith("._")
            )
        self.auto_workers = workers is None
        self.workers = workers or reference_workers(plan)
        self.core_deadline, self.interval = core_deadline, interval
        self.lock = threading.RLock()
        self.costs = ContinuationCosts(
            self.tasks,
            metadata=lambda identifier: read_cost_metadata(self.root, self.tasks[identifier]),
            measurements=load_pilot_measurements(plan["budget"].get("pilot_dir")),
            baseline_seconds_by_kind=plan["budget"].get("estimate", {}).get("serial_seconds_by_kind", {}),
        )
        self.costs.bootstrap(self.index)
        self.stop_event = threading.Event()
        self.thread = None

    def add_tasks(self, tasks):
        with self.lock:
            self.tasks.update({t["id"]: t for t in tasks})

    def verified(self, identifiers):
        with self.lock:
            self.index.complete.update(identifiers)

    def event(self, identifier, state):
        with self.lock:
            self.index.event(identifier, state)
            if state == "complete" and identifier in self.tasks:
                self.costs.observe_wall(self.tasks[identifier], self.index.durations.get(identifier, 0))

    def observe(self, task, result):
        with self.lock:
            self.costs.observe(task, result)

    def snapshot(self, *, refresh=False):
        with self.lock:
            if refresh:
                before = set(self.index.finished)
                self.index.refresh()
                for identifier in self.index.finished.keys() - before:
                    if identifier in self.tasks:
                        self.costs.observe_wall(self.tasks[identifier], self.index.durations.get(identifier, 0))
                fresh_micro = [
                    i
                    for i in self.index.finished.keys() - before
                    if i in self.tasks and self.tasks[i]["kind"] == "micro_probe" and self.index.durations.get(i, 0) > 1
                ]
                for identifier in sorted(fresh_micro, key=lambda i: self.index.finished[i])[-24:]:
                    self.costs.observe(self.tasks[identifier], self.costs.read(identifier))
            done = self.index.complete & self.tasks.keys()
            active = {k: v for k, v in self.index.active.items() if k in self.tasks and k not in done}
            ledger_path = self.root / "wall_budget.json"
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
            execution_path = self.root / "execution.json"
            execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
            if refresh and self.auto_workers:
                self.workers = execution.get("workers", self.workers)
            if refresh and ledger and not ledger["active"]:
                active = {}
            now = time.time()
            forecast = remaining_work(self.tasks, done, active, self.costs.predict, self.workers, now=now)
            used = (ledger["used"] + (max(0, now - ledger["started_wall"]) if ledger["active"] else 0)) if ledger else 0
            remaining = max(0, ledger["seconds"] - used) if ledger else None
            core_left = max(0, self.core_deadline - time.monotonic()) if self.core_deadline is not None else None
            if refresh and ledger:
                reserve = execution.get(
                    "core_reserve_seconds",
                    self.plan["budget"].get("analysis_reserve_seconds", 0)
                    + self.plan["budget"].get("followup_reserve_seconds", 0),
                )
                core_left = max(0, remaining - reserve)
            groups: dict[str, dict[str, int]] = defaultdict(lambda: dict(completed=0, planned=0))
            for identifier, task in self.tasks.items():
                group = groups[task["kind"]]
                group["planned"] += 1
                group["completed"] += identifier in done
            return dict(
                directory=str(self.root.resolve()),
                plan_fingerprint=self.plan["plan_fingerprint"],
                timestamp=now,
                completed=len(done),
                planned=len(self.tasks),
                percent=100 * len(done) / max(1, len(self.tasks)),
                active=len(active),
                active_groups=dict(Counter(family(self.tasks[i]) for i in active)),
                failed=len(self.index.failed - done),
                deferred=len(self.index.deferred - done),
                workers=self.workers,
                groups=dict(groups),
                **forecast,
                estimate_quality="rough: native rates/original mother futures; pilot fallback for unobserved strata",
                cumulative_seconds=used,
                wall_budget_remaining_seconds=remaining,
                core_dispatch_remaining_seconds=core_left,
                exceeds_budget=(core_left if core_left is not None else remaining) is not None
                and forecast["known_remaining_seconds"] > (core_left if core_left is not None else remaining),
                scope="frozen tasks only; adaptive tasks enter when their follow-up plan is frozen",
            )

    def emit(self, *, persist=False, refresh=False):
        snapshot = self.snapshot(refresh=refresh)
        eta = (
            duration(snapshot["estimated_remaining_seconds"])
            if snapshot["estimated_remaining_seconds"] is not None
            else "unknown"
        )
        budget = snapshot["core_dispatch_remaining_seconds"]
        budget_text = f" | dispatch budget {duration(budget)}" if budget is not None else ""
        groups = snapshot["groups"]
        coverage = "".join(
            f" | {kind} {groups[kind]['completed']}/{groups[kind]['planned']}"
            for kind in ("micro_probe", "meso_probe")
            if kind in groups
        )
        print(
            f"[overall] {snapshot['completed']}/{snapshot['planned']} ({snapshot['percent']:.1f}%)"
            f" | active {snapshot['active']} | failed {snapshot['failed']} | deferred {snapshot['deferred']}"
            f" | remaining ~{eta}{budget_text}{coverage}"
            + (" | OVER BUDGET" if snapshot["exceeds_budget"] else "")
            + (" | active task exceeded estimate" if snapshot["active_estimate_overruns"] else ""),
            flush=True,
        )
        if persist:
            atomic_json(self.control / "progress.json", snapshot)
        return snapshot

    def __enter__(self):
        try:
            self.emit(persist=True)
        except Exception as error:
            print(f"[overall] initial progress unavailable: {error}", file=sys.stderr, flush=True)

        def heartbeat():
            while not self.stop_event.wait(self.interval):
                try:
                    self.emit(persist=True)
                except Exception as error:
                    print(f"[overall] progress unavailable: {error}", file=sys.stderr, flush=True)

        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.emit(persist=True)
        except Exception as error:
            print(f"[overall] final progress unavailable: {error}", file=sys.stderr, flush=True)


def inspect_progress(root, *, watch=False, interval=15):
    if interval <= 0:
        raise ValueError("positive progress interval required")
    plan = read_plan(Path(root) / "plan.json")
    progress = OverallProgress(root, plan, interval=interval)
    try:
        if not watch:
            return progress.snapshot(refresh=True)
        while True:
            progress.emit(refresh=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        return None
