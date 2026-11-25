"""Isolated benchmark adapters; these do not replace production analysis storage.

All writers use WAL and synchronous=FULL. TinyMongo's per-connection FULL setup
uses its private backend hook solely for this controlled comparison. General
projection there still reads and parses embedded JSON before discarding payload.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import time
import zlib
from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np

from common.storage.documents import DocumentStore, decode_document, encode_document

SCALAR_TYPES = {
    "origin": "TEXT",
    "mode": "TEXT",
    "analysis_id": "TEXT",
    "recsys_type": "TEXT",
    "decay": "REAL",
    "rewiring": "REAL",
    "repost_rate": "REAL",
    "complete": "INTEGER",
    "stale": "INTEGER",
}
INDEX_FIELDS = tuple(SCALAR_TYPES)
_OPERATORS = {"$eq", "$in", "$gt", "$gte", "$lt", "$lte"}


def _validate_document(document):
    if not isinstance(document, Mapping) or not isinstance(document.get("_id"), str):
        raise ValueError("benchmark documents require a string _id")
    payload = document.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping with NumPy arrays and JSON values")
    if any(value.dtype.hasobject for value in _arrays(payload)):
        raise ValueError("object arrays are not supported")


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _scalar_state(state):
    if not isinstance(state, Mapping) or "_id" in state or "payload" in state:
        raise ValueError("state must be a mapping of fields other than _id/payload")
    if any(not isinstance(key, str) or "." in key or key.startswith("$") for key in state):
        raise ValueError("state updates require literal top-level field names")
    if any(isinstance(value, (Mapping, list, tuple, np.ndarray)) for value in state.values()):
        raise ValueError("state update values must be scalars")
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in state.items()}


def _lookup(document, field):
    value = document
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _matches(document, filters):
    for field, expected in filters.items():
        actual = _lookup(document, field)
        operations = expected if isinstance(expected, Mapping) else {"$eq": expected}
        for operator, operand in operations.items():
            if operator not in _OPERATORS:
                raise ValueError(f"unsupported benchmark filter operator: {operator}")
            if operator == "$eq" and actual != operand:
                return False
            if operator == "$in" and actual not in operand:
                return False
            if operator in ("$gt", "$gte", "$lt", "$lte"):
                if actual is None:
                    return False
                if operator == "$gt" and not actual > operand:
                    return False
                if operator == "$gte" and not actual >= operand:
                    return False
                if operator == "$lt" and not actual < operand:
                    return False
                if operator == "$lte" and not actual <= operand:
                    return False
    return True


def _project(document):
    return {key: value for key, value in document.items() if key != "payload"}


def _array_blob(array):
    with io.BytesIO() as stream:
        np.save(stream, array, allow_pickle=False)
        return zlib.compress(stream.getvalue())


def _array_from_blob(blob):
    with io.BytesIO(zlib.decompress(blob)) as stream:
        return np.load(stream, allow_pickle=False)


def _split_payload(payload):
    blobs = []

    def visit(value, path):
        if isinstance(value, np.ndarray):
            key = json.dumps(path, separators=(",", ":"))
            blobs.append((key, _array_blob(value)))
            return {"type": "array", "key": key}
        if isinstance(value, Mapping):
            return {
                "type": "mapping",
                "items": [[str(key), visit(item, [*path, str(key)])] for key, item in value.items()],
            }
        if isinstance(value, (list, tuple)):
            return {"type": "list", "items": [visit(item, [*path, index]) for index, item in enumerate(value)]}
        return {"type": "value", "value": encode_document(value)}

    layout = json.dumps(visit(payload, []), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return layout, blobs


def _join_payload(layout, arrays):
    def visit(node):
        kind = node["type"]
        if kind == "array":
            return arrays[node["key"]]
        if kind == "mapping":
            return {key: visit(item) for key, item in node["items"]}
        if kind == "list":
            return [visit(item) for item in node["items"]]
        if kind == "value":
            return decode_document(node["value"])
        raise ValueError(f"invalid payload layout kind: {kind}")

    return visit(json.loads(layout))


def _pragma_values(connection):
    return {
        "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0].lower(),
        "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
        "busy_timeout": connection.execute("PRAGMA busy_timeout").fetchone()[0],
    }


class _Backend:
    def __init__(self, directory, readonly=False):
        self.directory = Path(directory).resolve()
        self.readonly = readonly
        self.index_setup_seconds = 0.0
        self._before_commit = None
        self.counters = {"payload_array_decodes": 0, "payload_blob_reads": 0}

    def _writable(self):
        if self.readonly:
            raise RuntimeError("benchmark backend is read-only")

    @property
    def recovery_before_commit(self):
        return self._before_commit

    @recovery_before_commit.setter
    def recovery_before_commit(self, hook):
        self._before_commit = hook
        if hasattr(self, "connection"):
            self._install_recovery_trace(self.connection)

    def _install_recovery_trace(self, connection):
        if self._before_commit is None:
            connection.set_trace_callback(None)
            return
        changed = False

        def trace(statement):
            nonlocal changed
            sql = statement.lstrip().upper()
            # Exclude TinyMongo's catalog/index maintenance commits. The callback
            # fires before COMMIT executes, after real document DML was observed.
            if re.match(r"(?:INSERT(?: OR \w+)? INTO|UPDATE|DELETE FROM)\s+[\"\[]?(?:RECORDS|DOCUMENTS)\b", sql):
                changed = True
            if sql.startswith("COMMIT") and changed:
                changed = False
                self._before_commit()
            elif sql.startswith("ROLLBACK"):
                changed = False

        connection.set_trace_callback(trace)

    def _count_decodes(self, document):
        if document is not None:
            self.counters["payload_array_decodes"] += sum(1 for _ in _arrays(document.get("payload", {})))
        return document

    def _verify_durability(self):
        settings = self.pragmas()
        if settings["journal_mode"] != "wal" or settings["synchronous"] != 2:
            raise RuntimeError(f"benchmark requires WAL/FULL, observed {settings}")

    def checkpoint(self):
        self._writable()
        connection = self._diagnostic_connection()
        try:
            busy, log, checkpointed = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            return {"busy": busy, "log_frames": log, "checkpointed_frames": checkpointed}
        finally:
            if connection is not getattr(self, "connection", None):
                connection.close()

    def pragmas(self):
        connection = self._diagnostic_connection()
        try:
            values = _pragma_values(connection)
        finally:
            if connection is not getattr(self, "connection", None):
                connection.close()
        return {
            **values,
            "sqlite_version": sqlite3.sqlite_version,
            "index_setup_seconds": self.index_setup_seconds,
            "configuration": "explicit WAL and synchronous=FULL",
        }

    def integrity_check(self):
        connection = self._diagnostic_connection()
        try:
            return [row[0] for row in connection.execute("PRAGMA integrity_check")]
        finally:
            if connection is not getattr(self, "connection", None):
                connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CurrentBackend(_Backend):
    """Exercise the current production DocumentStore and its eager array codec."""

    namespace = "storage-benchmark"

    def __init__(self, directory, readonly=False):
        super().__init__(directory, readonly)
        self.store = DocumentStore(self.directory / "current.sqlite", readonly=readonly)
        self.path = self.store.path
        self.connection = self.store.connection
        self.connection.execute("PRAGMA synchronous=FULL")
        self._verify_durability()

    def _diagnostic_connection(self):
        return self.connection

    def put(self, document):
        self._writable()
        _validate_document(document)
        self.store.put(self.namespace, document["_id"], dict(document))

    def get(self, identifier, include_payload=True):
        document = self._count_decodes(self.store.get(self.namespace, identifier))
        return _project(document) if document is not None and not include_payload else document

    def query(self, filters, projection=False):
        documents = []
        for document in self.store.values(self.namespace):
            self._count_decodes(document)
            if _matches(document, filters):
                documents.append(_project(document) if projection else document)
        return documents

    def update_state(self, identifier, state):
        self._writable()
        state = _scalar_state(state)
        document = self.get(identifier)
        if document is None:
            raise KeyError(identifier)
        document.update(state)
        self.put(document)

    def count(self):
        return self.connection.execute(
            "SELECT COUNT(*) FROM documents WHERE namespace=?", (self.namespace,)
        ).fetchone()[0]

    def close(self):
        self.store.close()


class TinyMongoBackend(_Backend):
    """Native stable TinyMongo SQLite API, with benchmark-only FULL injection."""

    def __init__(self, directory, readonly=False):
        super().__init__(directory, readonly)
        from tinymongo import TinyMongoClient

        self.package_version = version("tinymongo")
        if self.package_version != "1.3.1":
            raise RuntimeError(f"benchmark requires tinymongo==1.3.1, found {self.package_version}")
        self.path = self.directory / "benchmark.sqlite"
        if readonly and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.client = TinyMongoClient(str(self.directory), backend="sqlite")
        self.database = self.client["benchmark"]
        self.engine = self.database.engine
        self.original_synchronous_values = set()
        original_connect = self.engine._read_connect

        def full_connect(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            self.original_synchronous_values.add(connection.execute("PRAGMA synchronous").fetchone()[0])
            connection.execute("PRAGMA synchronous=FULL")
            self._install_recovery_trace(connection)
            return connection

        self.engine._read_connect = full_connect
        self.collection = self.database["records"]
        if not readonly:
            started = time.perf_counter()
            for field in INDEX_FIELDS:
                self.collection.create_index([(field, 1)])
            self.index_setup_seconds = time.perf_counter() - started
        self._verify_durability()

    def _diagnostic_connection(self):
        return self.engine._connect()

    def put(self, document):
        self._writable()
        _validate_document(document)
        encoded = encode_document(dict(document))
        # This benchmark has one writer. Public insert_many([doc]) keeps one
        # commit and full validation, but avoids replace_one(upsert=True)'s
        # collection-wide post-image scan for each new ID in TinyMongo 1.3.1.
        # The pre-read/insert pair is not an atomic concurrent-writer upsert.
        existing = self.collection.find_one({"_id": document["_id"]}, projection={"_id": 1})
        if existing is None:
            self.collection.insert_many([encoded], ordered=True)
        else:
            self.collection.replace_one({"_id": document["_id"]}, encoded, upsert=True)

    def get(self, identifier, include_payload=True):
        projection = None if include_payload else {"payload": 0}
        document = self.collection.find_one({"_id": identifier}, projection=projection)
        if document is None:
            return None
        return self._count_decodes(decode_document(document))

    def query(self, filters, projection=False):
        columns = {"payload": 0} if projection else None
        return [self._count_decodes(decode_document(row)) for row in self.collection.find(filters, projection=columns)]

    def update_state(self, identifier, state):
        self._writable()
        state = _scalar_state(state)
        result = self.collection.update_one({"_id": identifier}, {"$set": encode_document(state)})
        if not result.matched_count:
            raise KeyError(identifier)

    def count(self):
        return self.collection.count_documents({})

    def pragmas(self):
        values = super().pragmas()
        return {
            **values,
            "configuration": "benchmark-only instance override of SQLiteTableBackend._read_connect; FULL on every connection",
            "tinymongo_version": self.package_version,
            "unmodified_connection_synchronous_values": sorted(self.original_synchronous_values),
            "readonly_contract": "adapter rejects mutations; TinyMongo initialization/read planning may write its catalog",
            "projection_contract": "ordinary projections parse full embedded JSON before discarding payload; NPY decode follows projection",
        }

    def close(self):
        self.client.close()


class IndexedSplitBackend(_Backend):
    """Indexed metadata and compressed NPY BLOBs in one SQLite transaction."""

    def __init__(self, directory, readonly=False):
        super().__init__(directory, readonly)
        self.path = self.directory / "indexed_split.sqlite"
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            columns = ",".join(f'"{field}" {kind}' for field, kind in SCALAR_TYPES.items())
            with self.connection:
                self.connection.execute(
                    f"CREATE TABLE IF NOT EXISTS records (_id TEXT PRIMARY KEY,document_json TEXT NOT NULL,"
                    f"payload_layout TEXT NOT NULL,{columns})"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS payloads (document_id TEXT NOT NULL,name TEXT NOT NULL,"
                    "data BLOB NOT NULL,PRIMARY KEY(document_id,name),"
                    "FOREIGN KEY(document_id) REFERENCES records(_id) ON DELETE CASCADE)"
                )
            started = time.perf_counter()
            with self.connection:
                for field in INDEX_FIELDS:
                    self.connection.execute(f'CREATE INDEX IF NOT EXISTS "idx_{field}" ON records("{field}")')
            self.index_setup_seconds = time.perf_counter() - started
        self._verify_durability()

    def _diagnostic_connection(self):
        return self.connection

    @staticmethod
    def _header_values(document):
        header = _project(document)
        encoded = json.dumps(encode_document(header), sort_keys=True, separators=(",", ":"), allow_nan=False)
        scalars = [document.get(field) for field in SCALAR_TYPES]
        scalars = [value.item() if isinstance(value, np.generic) else value for value in scalars]
        return header, encoded, scalars

    def put(self, document):
        self._writable()
        _validate_document(document)
        _header, encoded, scalars = self._header_values(document)
        layout, blobs = _split_payload(document.get("payload", {}))
        payloads = [(document["_id"], name, blob) for name, blob in blobs]
        fields = ",".join(f'"{field}"' for field in SCALAR_TYPES)
        updates = ",".join(f'"{field}"=excluded."{field}"' for field in SCALAR_TYPES)
        marks = ",".join("?" for _ in range(3 + len(SCALAR_TYPES)))
        with self.connection:
            self.connection.execute(
                f"INSERT INTO records(_id,document_json,payload_layout,{fields}) VALUES({marks}) "
                f"ON CONFLICT(_id) DO UPDATE SET document_json=excluded.document_json,"
                f"payload_layout=excluded.payload_layout,{updates}",
                [document["_id"], encoded, layout, *scalars],
            )
            self.connection.execute("DELETE FROM payloads WHERE document_id=?", (document["_id"],))
            self.connection.executemany("INSERT INTO payloads(document_id,name,data) VALUES(?,?,?)", payloads)

    def _load_payload(self, identifier):
        layout = self.connection.execute("SELECT payload_layout FROM records WHERE _id=?", (identifier,)).fetchone()[0]
        arrays = {}
        for name, blob in self.connection.execute(
            "SELECT name,data FROM payloads WHERE document_id=? ORDER BY name", (identifier,)
        ):
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
        row = self.connection.execute("SELECT document_json FROM records WHERE _id=?", (identifier,)).fetchone()
        if row is None:
            return None
        document = decode_document(json.loads(row[0]))
        if include_payload:
            document["payload"] = self._load_payload(identifier)
        return document

    @staticmethod
    def _compile_filters(filters):
        clauses, parameters = [], []
        sql_operators = {"$eq": "=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        for field, expected in filters.items():
            if not isinstance(field, str) or any(not part or part.startswith("$") for part in field.split(".")):
                raise ValueError("benchmark filters require literal field paths")
            path = None
            if field in SCALAR_TYPES or field == "_id":
                column = f'"{field}"'
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
        rows = self.connection.execute("SELECT _id,document_json FROM records" + where, parameters).fetchall()
        documents = []
        for identifier, raw in rows:
            document = decode_document(json.loads(raw))
            if not projection:
                document["payload"] = self._load_payload(identifier)
            documents.append(document)
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
            _header, encoded, scalars = self._header_values(document)
            fields = ",".join(f'"{field}"=?' for field in SCALAR_TYPES)
            self.connection.execute(
                f"UPDATE records SET document_json=?,{fields} WHERE _id=?", [encoded, *scalars, identifier]
            )

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def close(self):
        self.connection.close()


def open_backend(name, directory, readonly=False):
    """Open an isolated benchmark backend without mutating production configuration."""
    backends = {"current": CurrentBackend, "tinymongo": TinyMongoBackend, "indexed_split": IndexedSplitBackend}
    if name not in backends:
        raise ValueError(f"unknown storage benchmark backend: {name}")
    return backends[name](directory, readonly=readonly)
