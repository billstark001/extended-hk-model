"""CSV aggregation and raw-evidence readers for validation manuscript reports."""

from __future__ import annotations

import collections
import csv
import gzip
import json

import numpy as np


def read_csv(root, family, name):
    """Read one validation family analysis table below a frozen run root."""
    with (root / family / "analysis" / name).open() as f:
        return list(csv.DictReader(f))


def number(row, key):
    """Parse a scalar table field, returning None for absent, blank or literal None values."""
    return float(row[key]) if row.get(key) not in (None, "", "None") else None


def write_csv(path, rows):
    """Write heterogeneous report rows using the sorted union of columns; the parent directory must exist."""
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, metrics):
    """Group CSV records by keys and return count/min/median/max for numeric metrics, omitting missing values."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    result = []
    for group, items in sorted(grouped.items()):
        row = dict(zip(keys, group))
        row["cases"] = len(items)
        for metric in metrics:
            values = [number(r, metric) for r in items]
            values = [v for v in values if v is not None]
            if values:
                for name, value in zip(("min", "median", "max"), np.percentile(values, (0, 50, 100))):
                    row[metric + "_" + name] = float(value)
        result.append(row)
    return result


def raw(root, family, identity):
    """Load one identified gzip raw record from a validation family; never alter recorded evidence."""
    with gzip.open(root / family / "raw" / (identity + ".json.gz"), "rt") as f:
        return json.load(f)


def energy_only_tail(record):
    """Return (passes, stable_steps) for the historical saved-frame energy criterion, independently of state convergence."""
    stop = record["job"]["payload"]["request"]["stopping"]
    frames = record["result"]["frames"]
    previous = frames[0]
    stable = 0
    for frame in frames[1:]:
        if frame["step"] % stop["check_every"]:
            continue
        elapsed = frame["step"] - previous["step"]
        ok = all(
            abs(frame[k] - previous[k]) / elapsed
            <= stop["energy_absolute_tolerance"]
            + stop["energy_relative_tolerance"] * max(abs(frame[k]), abs(previous[k]))
            for k in ("node_energy", "edge_energy")
        )
        stable = stable + elapsed if ok and frame["step"] >= stop["minimum_steps"] else 0
        previous = frame
    return stable >= stop["patience_steps"], stable


def clock_group(row):
    """Return the physical/kernel/grid grouping key used for validation-clock crossing comparisons."""
    return tuple(row[k] for k in ("dynamics", "recommender", "zeta", "epsilon", "alpha", "b", "scan"))
