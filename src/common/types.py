"""Model-independent storage/provenance contracts; no runtime data conversion.

TypedDict instances remain plain dictionaries. Generic codecs deliberately keep
unknown payload values at their decoding boundary; scientific readers supply
more precise task/result types without copying arrays.
"""

from collections.abc import Callable
from os import PathLike
from typing import Generic, Literal, TypeAlias, TypeVar

from typing_extensions import NotRequired, TypedDict

PathInput: TypeAlias = str | PathLike[str]
Backend: TypeAlias = Literal["purego", "accelerate", "openblas"]
FileHashes: TypeAlias = dict[str, str]
TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


class Record(TypedDict, Generic[TaskT, ResultT]):
    """Decoded legacy/native envelope; the task identity is never regenerated."""

    task: TaskT
    result: ResultT


class RecordHeader(TypedDict, Generic[TaskT]):
    format: Literal["ehk-record"]
    schema: Literal[2]
    task: TaskT


class RecordFooter(TypedDict):
    kind: Literal["complete"]
    sha256: str


class TransportSuccess(TypedDict, Generic[ResultT]):
    id: str
    status: Literal["ok"]
    result: ResultT


class TransportFailure(TypedDict):
    id: str
    status: Literal["error"]
    error: str


class StorageOwner(TypedDict):
    schema: Literal[1]
    root: str
    control: str


class PendingCommit(TypedDict):
    destination: str
    source: str
    sha256: str | None


EventState: TypeAlias = Literal[
    "running", "complete", "retrying", "failed", "waiting_for_storage", "record_committed", "partial_analysis_failed"
]
EventHook: TypeAlias = Callable[..., None]


class EventDetails(TypedDict, total=False):
    attempt: int
    error: str
    path: str
    sha256: str


class RunEvent(EventDetails):
    time: float
    task: str | None
    state: EventState


class RuntimeBinding(TypedDict):
    binary_sha256: str
    request_schema_sha256: str


class ScientificSources(TypedDict):
    meso: FileHashes
    micro: FileHashes
    engine: FileHashes


class BuildInputs(ScientificSources):
    backend: Backend
    go_version: str


class BuildManifest(BuildInputs):
    source_fingerprint: str
    binary: str
    binary_sha256: str
    cached_binary: NotRequired[str]
    meso_git_head: str
    micro_git_head: str
    module_sha256: str
    sum_sha256: str
