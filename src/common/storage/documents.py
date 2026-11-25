"""Transactional SQLite documents with typed, pickle-free NumPy values."""

from __future__ import annotations

import base64
import io
import json
import math
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np


def encode_document(value):
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
        if self.readonly:
            raise RuntimeError("document store is read-only")
        body = json.dumps(encode_document(document), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.connection:
            self.connection.execute(
                "INSERT INTO documents(namespace,key,body,updated_ns) VALUES(?,?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET body=excluded.body,updated_ns=excluded.updated_ns",
                (namespace, key, body, time.time_ns()),
            )

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
