"""Stream preserved exposure feeds through one typed decoding boundary."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from smp_bindings.domain import SavedPost, TraceReceipt, parse_dataclass


@dataclass(kw_only=True, slots=True)
class PostPlacement:
    holder: int
    round: int


@dataclass(kw_only=True, slots=True)
class FeedSlot:
    post: SavedPost | None
    placement: PostPlacement | None
    source: str


@dataclass(kw_only=True, slots=True)
class ExposureFeedRow:
    step: int
    recipient: int
    opinion: float
    requested: int
    neighbors: tuple[FeedSlot, ...]
    recommendations: tuple[FeedSlot, ...]


def iter_exposure_trace(dataset: Path, receipt: TraceReceipt):
    """Decode each saved feed once; verify its receipt when exhausted."""
    if receipt.version != 1:
        raise ValueError("unsupported exposure trace version")
    updates = slots = 0
    with gzip.open(Path(dataset) / receipt.path, "rt", encoding="utf8") as source:
        for line in source:
            row = parse_dataclass(ExposureFeedRow, json.loads(line))
            updates += 1
            slots += len(row.recommendations)
            yield row
    if updates != receipt.updates or slots != receipt.slots:
        raise ValueError("exposure trace differs from its receipt")
