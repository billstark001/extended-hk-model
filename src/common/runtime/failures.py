"""Distinguish global protocol violations from independent scientific failures."""


class ProtocolError(ValueError):
    """A frozen identity or global invariant changed; stop the batch immediately."""


class TaskError(RuntimeError):
    """One independent scientific task failed; retain its diagnostic."""


class TaskDeferred(RuntimeError):
    """Operational admission/budget stop; no completed scientific result exists."""


class BatchTaskError(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__(f"{len(failures)} tasks failed; completed tasks retained: {failures}")
