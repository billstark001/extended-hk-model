"""Write heterogeneous rows without dropping fields."""

import csv


def table(path, rows):
    """Write CSV rows using the sorted union of their keys, creating parents; missing fields are blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
