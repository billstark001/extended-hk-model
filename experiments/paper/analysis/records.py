"""Versioned analysis documents, independent of the archived ORM statistics."""

import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from common.io.json import fingerprint
from common.storage.documents import DocumentStore

STATISTIC_FIELDS = (
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "rho_edge",
    "rho_post",
    "pathway_label",
    "separation_step",
    "segregation_step",
    "pathway_integral",
    "event_count",
    "event_step_mean",
    "triads",
    "x_bc_hom",
    "bc_hom_smpl",
    "x_mean_vars",
    "mean_vars_smpl",
    "opinion_diff",
    "x_opinion_diff_mean",
    "opinion_diff_mean_smpl",
    "last_community_count",
    "last_community_sizes",
    "last_opinion_peak_count",
    "p_backdrop",
    "h_backdrop",
    "g_backdrop",
    "opinion_diff_seg_mean",
    "opinion_diff_seg_std",
)


class ScenarioStatistics(SimpleNamespace):
    """Readable record attributes; new statistics need no database migration."""

    def __init__(self, **values):
        super().__init__(**{name: None for name in STATISTIC_FIELDS})
        self.__dict__.update(values)


class StatisticsStore:
    namespace = "ehk-paper-statistics-v3"

    def __init__(self, path, *, readonly=False):
        path = Path(path).resolve()
        if path.is_file():
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                legacy = database.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scenariostatistics'"
                ).fetchone()
            if legacy:
                raise ValueError("archived wide-table statistics require a separate new document database")
        self.documents = DocumentStore(path, readonly=readonly)

    @staticmethod
    def _key(name, origin, analysis_id):
        return fingerprint([analysis_id, origin, name])

    def get(self, name, origin, analysis_id):
        document = self.documents.get(self.namespace, self._key(name, origin, analysis_id))
        return None if document is None else ScenarioStatistics(**document)

    def put(self, record, *, analysis_id, mode, metadata=None, protocol=None, source=None):
        document = {
            **vars(record),
            "schema": 3,
            "analysis_id": analysis_id,
            "mode": mode,
            "metadata": metadata or {},
            "protocol": protocol or {},
            "source": source or {},
        }
        self.documents.put(self.namespace, self._key(record.name, record.origin, analysis_id), document)

    def records(self, *, origin=None, mode=None, analysis_id=None):
        records = []
        for document in self.documents.values(self.namespace):
            if origin is not None and document["origin"] != origin:
                continue
            if mode is not None and document["mode"] != mode:
                continue
            if analysis_id is not None and document["analysis_id"] != analysis_id:
                continue
            records.append(ScenarioStatistics(**document))
        return sorted(records, key=lambda row: (row.origin, row.name, row.analysis_id))

    def invalidate(self, name, origin, analysis_id, *, reason, current_source=None):
        """Keep provenance visible while preventing stale results from looking complete."""
        record = self.get(name, origin, analysis_id)
        if record is None:
            return
        record.complete = False
        record.stale = True
        record.stale_reason = reason
        record.invalidated_by_source = current_source
        self.put(
            record,
            analysis_id=analysis_id,
            mode=record.mode,
            metadata=record.metadata,
            protocol=record.protocol,
            source=record.source,
        )

    def close(self):
        self.documents.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
