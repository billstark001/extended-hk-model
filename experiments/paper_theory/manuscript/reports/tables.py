"""Typed scalar helpers for CSV manuscript-evidence tables."""

from __future__ import annotations

import csv


def read_table(path):
    """Return saved CSV rows as field dictionaries."""
    with path.open() as f:
        return list(csv.DictReader(f))


def number(row, key):
    """Parse a numeric field, treating missing/blank/None cells as absent."""
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else None
