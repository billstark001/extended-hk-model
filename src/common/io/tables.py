"""Write heterogeneous rows without dropping fields."""

import csv
import math


def table(path, rows):
    """Write CSV rows using the sorted union of their keys, creating parents; missing fields are blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def numeric_rows(path):
    """Read CSV rows, parsing numeric cells and representing empty cells as NaN."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value) if value != "" else math.nan
            except (ValueError, TypeError):
                pass
    return rows
