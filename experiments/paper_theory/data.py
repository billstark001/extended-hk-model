"""CSV export for the retained manuscript analyses."""

import csv
from pathlib import Path


def write_csv(path, rows):
    """Write nonempty rows with their union of fields, creating parents; reject empty input."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
