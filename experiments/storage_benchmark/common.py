"""Content checks and measurements shared by isolated benchmark processes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import threading
from pathlib import Path

import numpy as np

SCALARS = (
    "origin",
    "mode",
    "analysis_id",
    "recsys_type",
    "decay",
    "rewiring",
    "repost_rate",
    "tolerance",
    "post_retain_count",
    "complete",
    "stale",
)


def header_document(document):
    return {key: value for key, value in document.items() if key != "payload"}


def document_digest(document):
    """Hash values and exact ndarray dtype/shape/bytes independently of storage codec."""
    digest = hashlib.sha256()

    def emit(value):
        if isinstance(value, np.ndarray):
            digest.update(b"array:")
            digest.update(json.dumps([value.dtype.str, value.shape]).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.generic):
            emit(value.item())
        elif isinstance(value, dict):
            digest.update(b"dict:")
            for key in sorted(value):
                emit(str(key))
                emit(value[key])
            digest.update(b"end-dict")
        elif isinstance(value, (list, tuple)):
            digest.update(b"list:")
            for item in value:
                emit(item)
            digest.update(b"end-list")
        elif isinstance(value, float) and not math.isfinite(value):
            digest.update(("float:" + repr(value)).encode())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
            digest.update(b";")

    emit(document)
    return digest.hexdigest()


def payload_size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(payload_size(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_size(item) for item in value)
    return 0


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def current_rss_bytes():
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss


class RSSSampler:
    """Sample the action window, separately from import/source-calibration peaks."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak = current_rss_bytes()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, current_rss_bytes())

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        self.peak = max(self.peak, current_rss_bytes())
        self.stop.set()
        self.thread.join()
        return self.peak


def directory_size(directory):
    files = [path for path in Path(directory).rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "apparent_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(getattr(path.stat(), "st_blocks", 0) * 512 for path in files),
    }


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    return {
        "operations": len(values),
        "seconds": float(values.sum()),
        "p50_ms": float(np.quantile(values, 0.5) * 1000),
        "p95_ms": float(np.quantile(values, 0.95) * 1000),
        "p99_ms": float(np.quantile(values, 0.99) * 1000),
        "operations_per_second": float(len(values) / values.sum()) if values.sum() else None,
    }
