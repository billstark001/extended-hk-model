"""Read saved node/edge fit coefficients and typed parameter-clock tables."""

import csv

from experiments.paper_theory.release_clocks.artifact_paths import SOURCE


def load_fits():
    """Return keyed Gompertz node and edge-mixture fit rows from the read-only scan input."""
    n = {}
    e = {}
    for filename, dest in [("low_parameter_fits.csv", n), ("edge_timescale_fits.csv", e)]:
        for r in csv.DictReader((SOURCE / filename).open()):
            if dest is n and (r["energy"] != "node_energy" or r["model"] != "gompertz_hazard_2"):
                continue
            k = (r["dynamics"], round(float(r["alpha"]), 12), float(r["epsilon"]), round(float(r["q_over_alpha"]), 9))
            dest[k] = r
    return (n, e)


def read_saved_table(path):
    """Read typed metric tables so figures can be redrawn without solver caches."""
    rows = []
    with path.open() as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key == "dynamics":
                    row[key] = value
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows
