"""Versioned, typed analysis documents independent of archived ORM statistics."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, fields, replace
from pathlib import Path

from numpy.typing import NDArray

from common.io.json import fingerprint
from common.storage.documents import DocumentStore
from ehk.domain import ScenarioMetadata, InputIdentity, parse_metadata, parse_dataclass
from .models import AnalysisProtocol


@dataclass(slots=True)
class ScenarioStatistics:
    name: str = ""
    origin: str = ""
    tolerance: float | None = None
    decay: float | None = None
    rewiring: float | None = None
    repost_rate: float | None = None
    recsys_type: str | None = None
    post_retain_count: int | None = None
    step: int | None = None
    active_step: int | None = None
    active_step_threshold: float | None = None
    g_index_mean_active: float | None = None
    x_indices: NDArray | None = None
    h_index: NDArray | None = None
    p_index: NDArray | None = None
    g_index: NDArray | None = None
    rho_edge: NDArray | None = None
    rho_post: NDArray | None = None
    pathway_label: str | None = None
    separation_step: int | None = None
    segregation_step: int | None = None
    pathway_integral: float | None = None
    event_count: int | None = None
    event_step_mean: float | None = None
    triads: int | None = None
    x_bc_hom: NDArray | None = None
    bc_hom_smpl: NDArray | None = None
    x_mean_vars: NDArray | None = None
    mean_vars_smpl: NDArray | None = None
    opinion_diff: NDArray | None = None
    x_opinion_diff_mean: NDArray | None = None
    opinion_diff_mean_smpl: NDArray | None = None
    last_community_count: int | None = None
    last_community_sizes: str | None = None
    last_opinion_peak_count: int | None = None
    p_backdrop: float | None = None
    h_backdrop: float | None = None
    g_backdrop: float | None = None
    opinion_diff_seg_mean: NDArray | None = None
    opinion_diff_seg_std: NDArray | None = None
    schema: int = 3
    analysis_id: str = ""
    mode: str = ""
    metadata: ScenarioMetadata | None = None
    recorded_metadata: ScenarioMetadata | None = None
    requested_metadata: ScenarioMetadata | None = None
    metadata_source: str | None = None
    metadata_provenance: str | None = None
    rng_provenance: str | None = None
    endpoint_complement: bool | None = None
    metric_revision: int | None = None
    task_fingerprint: str | None = None
    complete: bool = False
    stale: bool = False
    stale_reason: str | None = None
    invalidated_by_source: InputIdentity | None = None
    protocol: AnalysisProtocol | None = None
    source: InputIdentity | None = None

    @classmethod
    def from_wire(cls, value):
        """Validate decoded SQLite fields without adding unknown attributes."""
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown statistics fields: {sorted(unknown)}")
        decoded = dict(value)
        for name in ("metadata", "recorded_metadata", "requested_metadata"):
            decoded[name] = parse_metadata(decoded[name]) if decoded.get(name) else None
        for name in ("source", "invalidated_by_source"):
            decoded[name] = parse_dataclass(InputIdentity, decoded[name]) if decoded.get(name) else None
        decoded["protocol"] = AnalysisProtocol.from_wire(decoded["protocol"]) if decoded.get("protocol") else None
        return cls(**decoded)


class StatisticsStore:
    namespace = "ehk-paper-statistics-v3"

    def __init__(self, path, *, readonly=False):
        path = Path(path).resolve()
        if path.is_file():
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                archived = database.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scenariostatistics'"
                ).fetchone()
            if archived:
                raise ValueError("archived wide-table statistics require a separate new document database")
        self.documents = DocumentStore(path, readonly=readonly)

    @staticmethod
    def _key(name, origin, analysis_id):
        return fingerprint([analysis_id, origin, name])

    def get(self, name, origin, analysis_id) -> ScenarioStatistics | None:
        document = self.documents.get(self.namespace, self._key(name, origin, analysis_id))
        return None if document is None else ScenarioStatistics.from_wire(document)

    def put(self, record: ScenarioStatistics, *, analysis_id, mode, metadata=None, protocol=None, source=None):
        document = replace(record, schema=3, analysis_id=analysis_id, mode=mode,
                           metadata=metadata if metadata is not None else record.metadata,
                           protocol=protocol if protocol is not None else record.protocol,
                           source=source if source is not None else record.source)
        self.documents.put(self.namespace, self._key(record.name, record.origin, analysis_id), document)

    def records(self, *, origin=None, mode=None, analysis_id=None) -> list[ScenarioStatistics]:
        records = []
        for document in self.documents.values(self.namespace):
            row = ScenarioStatistics.from_wire(document)
            if origin is not None and row.origin != origin:
                continue
            if mode is not None and row.mode != mode:
                continue
            if analysis_id is not None and row.analysis_id != analysis_id:
                continue
            records.append(row)
        return sorted(records, key=lambda row: (row.origin, row.name, row.analysis_id))

    def invalidate(self, name, origin, analysis_id, *, reason, current_source=None):
        record = self.get(name, origin, analysis_id)
        if record is not None:
            self.put(replace(record, complete=False, stale=True, stale_reason=reason,
                             invalidated_by_source=current_source), analysis_id=analysis_id, mode=record.mode)

    def close(self):
        self.documents.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
