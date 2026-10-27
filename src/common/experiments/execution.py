"""One task lifecycle with pluggable model and record adapters."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from common.runtime.failures import BatchTaskError, ProtocolError, TaskDeferred
from common.runtime.process import run_tasks

from .progress import Progress
from .scheduling import ReadyTasks

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskAdapter(Generic[T, R]):
    identifier: Callable[[T], str]
    cached: Callable[[T], bool]
    compute: Callable[[T], R]
    save: Callable[[T, R], bool | None]
    cancel: Callable[[], None] | None = None
    batch: Callable[..., None] | None = None


def execute_ready_tasks(
    tasks,
    adapter,
    *,
    completed,
    key,
    workers=1,
    deadline=None,
    store=None,
    journal=None,
    progress=False,
    on_finish=None,
):
    """Reuse the task lifecycle while releasing children only after commit/cache validation."""
    queue = ReadyTasks(tasks, completed, key=key, deadline=deadline)

    def cached(task):
        value = adapter.cached(task)
        if value:
            queue.complete(adapter.identifier(task))
        return value

    def save(task, result):
        value = adapter.save(task, result)
        if value is False:
            queue.stop()
        else:
            queue.complete(adapter.identifier(task))
        return value

    def cancel():
        queue.stop()
        if adapter.cancel:
            adapter.cancel()

    streaming = TaskAdapter(adapter.identifier, cached, adapter.compute, save, cancel=cancel)
    try:
        execute_tasks(queue, streaming, workers=workers, store=store, journal=journal, progress=progress, stream=True)
    finally:
        if on_finish:
            on_finish(set(queue.done))
    return set(queue.done)


def execute_tasks(
    tasks: Iterable[T],
    adapter: TaskAdapter[T, R],
    *,
    workers=1,
    force=False,
    limit=None,
    store=None,
    journal=None,
    executor="thread",
    raise_failures=True,
    progress=True,
    stream=False,
) -> None:
    if workers < 1 or executor not in ("thread", "process") or (limit is not None and limit < 1):
        raise ValueError("positive workers/limit and thread/process executor required")
    if stream and (limit is not None or executor != "thread" or adapter.batch is not None or progress is True):
        raise ValueError("streaming requires a thread adapter, no limit and an explicit progress sink or False")
    selected: Any = tasks if stream else (list(tasks)[:limit] if limit else list(tasks))
    if not stream and len({adapter.identifier(t) for t in selected}) != len(selected):
        raise ProtocolError("duplicate task identifiers")
    meter = Progress(len(selected)) if progress is True else (progress or None)

    def transition(task, state, **details):
        identifier = adapter.identifier(task)
        if meter:
            meter.event(identifier, state)
        if store:
            store.event({"id": identifier}, state, **details)
        if journal:
            journal.event(identifier, state, **details)

    def cached(task):
        def check():
            return not force and adapter.cached(task)

        value = store.read(check) if store else check()
        if value:
            transition(task, "cached")
        return value

    def save(task, result):
        complete = adapter.save(task, result)
        transition(task, "deferred" if complete is False else "complete")

    def run(task):
        try:
            if not cached(task):
                save(task, adapter.compute(task))
        except TaskDeferred as error:
            if stream and adapter.cancel:
                adapter.cancel()
            transition(task, "deferred", reason=str(error))
        except BaseException:
            if stream and adapter.cancel:
                adapter.cancel()
            raise

    def scheduling_event(task, state, **details):
        if state != "complete":
            transition(task, state, **details)

    if adapter.batch is not None:
        pending = [task for task in selected if not cached(task)]
        expected = {adapter.identifier(task): task for task in pending}
        completed = set()
        failures = []

        def validate(task):
            identifier = adapter.identifier(task)
            if identifier not in expected or identifier in completed:
                raise ProtocolError("batch returned an undeclared or duplicate task")
            completed.add(identifier)

        def complete(task, result):
            validate(task)
            try:
                save(task, result)
            except ProtocolError:
                raise
            except Exception as error:
                failures.append(str(error))
                transition(task, "failed", error=str(error))

        def failed(task, error):
            if isinstance(error, ProtocolError):
                raise error
            validate(task)
            failures.append(str(error))
            transition(task, "failed", error=str(error))

        for task in pending:
            transition(task, "running")
        try:
            if pending:
                adapter.batch(pending, workers, complete, failed)
        except BaseException:
            if adapter.cancel:
                adapter.cancel()
            raise
        for identifier, task in expected.items():
            if identifier not in completed:
                failed(task, RuntimeError(f"batch did not return {identifier}"))
        if failures and raise_failures:
            raise BatchTaskError(failures)
        return

    if executor == "process":
        pending = [task for task in selected if not cached(task)]
        run_tasks(
            adapter.compute,
            pending,
            workers,
            isolate_failures=True,
            attempts=1,
            event=scheduling_event,
            on_result=save,
            executor="process",
            cancel=adapter.cancel,
            raise_failures=raise_failures,
        )
    else:
        run_tasks(
            run,
            selected,
            workers,
            isolate_failures=True,
            event=scheduling_event,
            cancel=adapter.cancel,
            raise_failures=raise_failures,
        )
