"""Transactional SQLite documents with typed, pickle-free NumPy values."""

from __future__ import annotations

import base64
import io
import json
import math
import sqlite3
import time
import zlib
from dataclasses import dataclass, fields, is_dataclass
from typing import Generic, TypeVar
from concurrent.futures import Future
from contextlib import closing
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

DocumentValue = TypeVar("DocumentValue")


@dataclass(frozen=True, slots=True)
class DocumentWrite(Generic[DocumentValue]):
    namespace: str
    key: str
    document: DocumentValue


@dataclass(frozen=True, slots=True)
class WriterRequest:
    operation: str
    arguments: tuple[object, ...]
    completion: Future


def encode_document(value):
    if hasattr(value, "__wire__"):
        return encode_document(value.__wire__(preserve_arrays=True))
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: encode_document(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, np.ndarray):
        with io.BytesIO() as stream:
            np.save(stream, value, allow_pickle=False)
            return {"$numpy_npy_zlib": base64.b64encode(zlib.compress(stream.getvalue())).decode("ascii")}
    if isinstance(value, np.generic):
        return encode_document(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {str(key): encode_document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_document(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def decode_document(value):
    if isinstance(value, dict):
        if set(value) == {"$numpy_npy_zlib"}:
            payload = zlib.decompress(base64.b64decode(value["$numpy_npy_zlib"], validate=True))
            with io.BytesIO(payload) as stream:
                return np.load(stream, allow_pickle=False)
        if set(value) == {"$nonfinite_float"}:
            return float(value["$nonfinite_float"])
        return {key: decode_document(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_document(item) for item in value]
    return value


class DocumentStore:
    """One atomic upsert per document; readers never create or migrate a database."""

    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        self.readonly = readonly
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS documents ("
                    "namespace TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL, "
                    "updated_ns INTEGER NOT NULL, PRIMARY KEY(namespace, key))"
                )

    def get(self, namespace, key):
        row = self.connection.execute(
            "SELECT body FROM documents WHERE namespace=? AND key=?", (namespace, key)
        ).fetchone()
        return None if row is None else decode_document(json.loads(row[0]))

    def put(self, namespace, key, document):
        self.put_many([DocumentWrite(namespace, key, document)])

    def put_many(self, documents):
        """Commit related documents together, including their completion index."""
        if self.readonly:
            raise RuntimeError("document store is read-only")
        rows = [
            (item.namespace, item.key, json.dumps(encode_document(item.document), sort_keys=True, separators=(",", ":"), allow_nan=False), time.time_ns())
            for item in documents
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT INTO documents(namespace,key,body,updated_ns) VALUES(?,?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET body=excluded.body,updated_ns=excluded.updated_ns",
                rows,
            )

    def backup(self, destination):
        """Create a consistent portable database, including committed WAL pages."""
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(destination)) as target:
            self.connection.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")

    def seal(self):
        """Finish a portable snapshot without sidecar journals or live writers."""
        if self.readonly:
            raise RuntimeError("document store is read-only")
        self.connection.execute("PRAGMA journal_mode=DELETE")

    def values(self, namespace):
        for (body,) in self.connection.execute(
            "SELECT body FROM documents WHERE namespace=? ORDER BY key", (namespace,)
        ):
            yield decode_document(json.loads(body))

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DocumentWriter:
    """One parent-process SQLite owner; workers submit bounded, acknowledged calls."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._queue = Queue(maxsize=32)
        self._ready = Future()
        self._closed = False
        self._thread = Thread(target=self._serve, name="sqlite-document-writer", daemon=True)
        self._thread.start()
        self._ready.result()

    def _serve(self):
        try:
            store = DocumentStore(self.path)
        except BaseException as error:
            self._ready.set_exception(error)
            return
        self._ready.set_result(None)
        try:
            while True:
                entry = self._queue.get()
                if entry is None:
                    break
                operation, args, future = entry.operation, entry.arguments, entry.completion
                try:
                    value = getattr(store, operation)(*args)
                    future.set_result(list(value) if operation == "values" else value)
                except BaseException as error:
                    future.set_exception(error)
        finally:
            store.close()

    def _call(self, operation, *args):
        if self._closed:
            raise RuntimeError("document writer is closed")
        future = Future()
        self._queue.put(WriterRequest(operation, args, future))
        return future.result()

    def get(self, namespace, key):
        return self._call("get", namespace, key)

    def values(self, namespace):
        return self._call("values", namespace)

    def put(self, namespace, key, document):
        return self._call("put", namespace, key, document)

    def put_many(self, documents):
        return self._call("put_many", list(documents))

    def backup(self, destination):
        return self._call("backup", destination)

    def seal(self):
        return self._call("seal")

    def close(self):
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
