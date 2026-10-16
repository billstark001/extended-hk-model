"""Offline readers and report orchestration for new measure-sweep caches."""

import json

import numpy as np


def cached_cases(directory):
    """Yield independent copies of NPZ case metadata, diagnostics and recorded arrays from directory/raw."""
    for path in sorted((directory / "raw").glob("*.npz")):
        if path.name.startswith("._"):
            continue  # macOS AppleDouble metadata on exFAT is not a result archive.
        with np.load(path, allow_pickle=False) as archive:
            item = json.loads(str(archive["metadata"]))
            p = item["parameters"]
            diagnostic = json.loads(str(archive["diagnostics"]))
            yield {
                **p,
                "r": p["ratio"],
                "converged": diagnostic["converged"],
                "source": path.name,
                "diagnostics": diagnostic,
                **{key: archive[key].copy() for key in archive.files if key not in ("metadata", "diagnostics")},
            }
