"""Canonical hashes and atomic JSON/gzip storage."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    """Serialize JSON with sorted keys and compact separators; reject NaN and infinities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    """Return the SHA256 of canonical JSON so identities do not depend on mapping insertion order."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    """Hash large records with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Create parents and replace a JSON file atomically after flushing its temporary file; reject nonfinite values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_result(path, value):
    """Create parents and atomically replace a gzip record containing canonical JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        with gzip.open(temp, "wt") as f:
            f.write(canonical(value))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_result(path):
    """Load one gzip-compressed JSON record; propagate decompression and JSON errors."""
    with gzip.open(path, "rt") as f:
        return json.load(f)
