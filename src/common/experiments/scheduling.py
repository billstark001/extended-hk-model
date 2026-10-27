"""Dependency-aware, work-conserving round-robin admission for thread adapters."""

import heapq
import threading
import time
from collections import defaultdict, deque
from typing import Any

from common.runtime.failures import ProtocolError


class ReadyTasks:
    """A stream whose workers acknowledge committed tasks before releasing children.

    The consumer may wait for a running worker to commit. Every worker failure
    must call stop(), so a missing dependency can never deadlock admission.
    """

    def __init__(self, tasks, completed, *, key, deadline=None, clock=time.monotonic):
        items = list(tasks)
        self.tasks = {t["id"]: t for t in items}
        if len(items) != len(self.tasks):
            raise ProtocolError("duplicate scheduler task identifiers")
        self.done = set(completed)
        self.pending = set(self.tasks) - self.done
        self.active = set()
        self.children = defaultdict(list)
        self.ready = defaultdict(list)
        self.lanes: deque[Any] = deque()
        self.key, self.deadline, self.clock = key, deadline, clock
        self.condition = threading.Condition()
        self.stopped = False
        for task in self.tasks.values():
            parent = task["source_task"]
            if parent and parent not in self.tasks and parent not in self.done:
                raise ProtocolError("scheduler dependency is outside the frozen task set")
            self.children[parent].append(task["id"])
        for identifier in sorted(self.pending):
            if not self.tasks[identifier]["source_task"] or self.tasks[identifier]["source_task"] in self.done:
                self._ready(identifier)

    def _ready(self, identifier):
        lane, priority = self.key(self.tasks[identifier])
        if not self.ready[lane]:
            self.lanes.append(lane)
        heapq.heappush(self.ready[lane], (priority, identifier))

    def complete(self, identifier):
        with self.condition:
            self.active.discard(identifier)
            self.done.add(identifier)
            for child in self.children[identifier]:
                if child in self.pending:
                    self._ready(child)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def __iter__(self):
        while True:
            with self.condition:
                while not self.lanes and self.active and not self.stopped:
                    if self.deadline is not None and self.clock() >= self.deadline:
                        self.stopped = True
                        break
                    self.condition.wait(0.25)
                if self.stopped or (self.deadline is not None and self.clock() >= self.deadline):
                    return
                if not self.pending:
                    return
                if not self.lanes:
                    raise ProtocolError("frozen dependency graph has a cycle or an uncommitted parent")
                lane = self.lanes.popleft()
                _, identifier = heapq.heappop(self.ready[lane])
                if self.ready[lane]:
                    self.lanes.append(lane)
                self.pending.remove(identifier)
                self.active.add(identifier)
            yield self.tasks[identifier]
