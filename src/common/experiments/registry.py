"""Named study tasks and explicitly declared aggregate selections."""

from typing import Generic, TypeVar

T = TypeVar("T")


class TaskRegistry(Generic[T]):
    def __init__(self, tasks: dict[str, T], groups: dict[str, tuple[str, ...]] | None = None):
        self.tasks = dict(tasks)
        self.groups = dict(groups or {})
        if set(self.tasks) & set(self.groups):
            raise ValueError("task and aggregate names must be distinct")
        for name, members in self.groups.items():
            if not members or len(set(members)) != len(members) or any(m not in self.tasks for m in members):
                raise ValueError(f"invalid aggregate selection: {name}")

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.groups, *self.tasks)

    def select(self, name: str) -> tuple[T, ...]:
        if name in self.groups:
            return tuple(self.tasks[m] for m in self.groups[name])
        if name in self.tasks:
            return (self.tasks[name],)
        raise ValueError(f"unknown task: {name}; choose from {', '.join(self.names)}")
