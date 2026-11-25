"""Isolated SQLite variant with streamed headers and integer-keyed array BLOBs.

This creates a new database; it neither migrates nor changes indexed_split. The
codec and recursive payload layout are shared with the frozen comparison backend.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

from common.storage.documents import decode_document, encode_document

from .backends import (
    _OPERATORS,
    SCALAR_TYPES,
    _array_from_blob,
    _Backend,
    _join_payload,
    _lookup,
    _project,
    _scalar_state,
    _split_payload,
    _validate_document,
)

COLUMN_TYPES = {**SCALAR_TYPES, "tolerance": "REAL", "post_retain_count": "INTEGER", "hk_influence": "REAL"}
FIELD_COLUMNS = {field: field for field in COLUMN_TYPES if field != "hk_influence"}
FIELD_COLUMNS.update({"_id": "_id", "metadata.HKParams.Influence": "hk_influence"})
COMPOSITE_INDEXES = {
    "idx_scope": ("origin", "mode", "analysis_id"),
    "idx_policy": (
        "origin",
        "mode",
        "analysis_id",
        "recsys_type",
        "decay",
        "rewiring",
        "repost_rate",
        "post_retain_count",
        "tolerance",
    ),
    "idx_influence": ("origin", "mode", "analysis_id", "hk_influence"),
}


class SQLiteOptimizedBackend(_Backend):
    """Same benchmark API, with one WAL/FULL transaction per document write."""

    def __init__(self, directory, readonly=False):
        super().__init__(directory, readonly)
        self.path = self.directory / "sqlite_optimized.sqlite"
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            columns = ",".join(f'"{field}" {kind}' for field, kind in COLUMN_TYPES.items())
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY,_id TEXT NOT NULL UNIQUE,"
                    f"document_json TEXT NOT NULL,payload_layout TEXT NOT NULL,{columns})"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS payloads (record_id INTEGER NOT NULL,name TEXT NOT NULL,"
                    "data BLOB NOT NULL,PRIMARY KEY(record_id,name),"
                    "FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE) WITHOUT ROWID"
                )
            started = time.perf_counter()
            with self.connection:
                for name, fields in COMPOSITE_INDEXES.items():
                    columns = ",".join(f'"{field}"' for field in fields)
                    self.connection.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON records({columns})')
            self.index_setup_seconds = time.perf_counter() - started
        self._verify_durability()

    def _diagnostic_connection(self):
        return self.connection

    @staticmethod
    def _header_values(document):
        encoded = json.dumps(
            encode_document(_project(document)), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        scalars = [
            _lookup(document, "metadata.HKParams.Influence") if field == "hk_influence" else document.get(field)
            for field in COLUMN_TYPES
        ]
        return encoded, [value.item() if isinstance(value, np.generic) else value for value in scalars]

    def put(self, document):
        self._writable()
        _validate_document(document)
        encoded, scalars = self._header_values(document)
        layout, blobs = _split_payload(document.get("payload", {}))
        fields = ",".join(f'"{field}"' for field in COLUMN_TYPES)
        updates = ",".join(f'"{field}"=excluded."{field}"' for field in COLUMN_TYPES)
        marks = ",".join("?" for _ in range(3 + len(COLUMN_TYPES)))
        with self.connection:
            record_id = self.connection.execute(
                f"INSERT INTO records(_id,document_json,payload_layout,{fields}) VALUES({marks}) "
                f"ON CONFLICT(_id) DO UPDATE SET document_json=excluded.document_json,"
                f"payload_layout=excluded.payload_layout,{updates} RETURNING id",
                [document["_id"], encoded, layout, *scalars],
            ).fetchone()[0]
            self.connection.execute("DELETE FROM payloads WHERE record_id=?", (record_id,))
            self.connection.executemany(
                "INSERT INTO payloads(record_id,name,data) VALUES(?,?,?)",
                ((record_id, name, blob) for name, blob in blobs),
            )

    def _load_payload(self, record_id, layout):
        arrays = {}
        for name, blob in self.connection.execute("SELECT name,data FROM payloads WHERE record_id=?", (record_id,)):
            arrays[name] = _array_from_blob(blob)
            self.counters["payload_blob_reads"] += 1
            self.counters["payload_array_decodes"] += 1
        return _join_payload(layout, arrays)

    @contextmanager
    def _read_snapshot(self):
        owner = not self.connection.in_transaction
        if owner:
            self.connection.execute("BEGIN")
        try:
            yield
            if owner:
                self.connection.commit()
        except BaseException:
            if owner:
                self.connection.rollback()
            raise

    def get(self, identifier, include_payload=True):
        if include_payload and not self.connection.in_transaction:
            with self._read_snapshot():
                return self.get(identifier, include_payload=True)
        columns = "document_json,id,payload_layout" if include_payload else "document_json"
        row = self.connection.execute(f"SELECT {columns} FROM records WHERE _id=?", (identifier,)).fetchone()
        if row is None:
            return None
        document = decode_document(json.loads(row[0]))
        if include_payload:
            document["payload"] = self._load_payload(row[1], row[2])
        return document

    @staticmethod
    def _compile_filters(filters):
        clauses, parameters = [], []
        sql_operators = {"$eq": "=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        for field, expected in filters.items():
            if not isinstance(field, str) or any(not part or part.startswith("$") for part in field.split(".")):
                raise ValueError("benchmark filters require literal field paths")
            path = None
            if field in FIELD_COLUMNS:
                column = f'"{FIELD_COLUMNS[field]}"'
            else:
                path = "$" + "".join("." + json.dumps(part) for part in field.split("."))
                column = "json_extract(document_json, ?)"
            operations = expected if isinstance(expected, Mapping) else {"$eq": expected}
            for operator, operand in operations.items():
                if operator not in _OPERATORS:
                    raise ValueError(f"unsupported benchmark filter operator: {operator}")
                if operator == "$in":
                    values = list(operand)
                    nonnull = [value for value in values if value is not None]
                    alternatives = []
                    if nonnull:
                        alternatives.append(f"{column} IN ({','.join('?' for _ in nonnull)})")
                        if path is not None:
                            parameters.append(path)
                        parameters.extend(nonnull)
                    if None in values:
                        alternatives.append(f"{column} IS NULL")
                        if path is not None:
                            parameters.append(path)
                    clauses.append("(" + " OR ".join(alternatives) + ")" if alternatives else "0")
                elif operator == "$eq" and operand is None:
                    clauses.append(f"{column} IS NULL")
                    if path is not None:
                        parameters.append(path)
                else:
                    clauses.append(f"{column} {sql_operators[operator]} ?")
                    if path is not None:
                        parameters.append(path)
                    parameters.append(operand)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters

    def query(self, filters, projection=False):
        if not projection and not self.connection.in_transaction:
            with self._read_snapshot():
                return self.query(filters, projection=False)
        where, parameters = self._compile_filters(filters)
        columns = "document_json" if projection else "document_json,id,payload_layout"
        cursor = self.connection.execute("SELECT " + columns + " FROM records" + where, parameters)
        documents = []
        try:
            for row in cursor:
                document = decode_document(json.loads(row[0]))
                if not projection:
                    document["payload"] = self._load_payload(row[1], row[2])
                documents.append(document)
        finally:
            cursor.close()
        return documents

    def update_state(self, identifier, state):
        self._writable()
        state = _scalar_state(state)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            document = self.get(identifier, include_payload=False)
            if document is None:
                raise KeyError(identifier)
            document.update(state)
            encoded, scalars = self._header_values(document)
            fields = ",".join(f'"{field}"=?' for field in COLUMN_TYPES)
            self.connection.execute(
                f"UPDATE records SET document_json=?,{fields} WHERE _id=?", [encoded, *scalars, identifier]
            )

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def pragmas(self):
        return {
            **super().pragmas(),
            "payload_storage": "integer foreign key; composite primary key WITHOUT ROWID; shared NPY/zlib codec",
            "query_materialization": "cursor iteration into the required result list; no raw-header fetchall",
            "indexes": {name: list(fields) for name, fields in COMPOSITE_INDEXES.items()},
        }

    def close(self):
        self.connection.close()
