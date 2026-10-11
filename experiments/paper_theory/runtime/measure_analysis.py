"""Offline readers and report orchestration for new measure-sweep caches."""

import json

import numpy as np

from experiments.paper_theory.data import write_csv

from ..release_clocks.analyze.means import summarize_case
from ..scaling.analyze import scaling_rows


def cached_cases(directory):
    """Yield independent copies of NPZ case metadata, diagnostics and recorded arrays from directory/raw."""
    for path in sorted((directory / "raw").glob("*.npz")):
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


def new_runs(root, suite, profile):
    """Write mean release clocks, compatible curve comparisons and conservation diagnostics for completed suite/profile caches."""
    directory = root / suite / profile
    cases = list(cached_cases(directory))
    if not cases:
        raise FileNotFoundError(f"no results in {directory / 'raw'}")
    rows = [summarize_case(c) for c in cases]
    write_csv(directory / "tables" / "mean_release.csv", rows)
    comparisons = scaling_rows(cases, population=suite == "population_closure")
    if comparisons:
        write_csv(directory / "tables" / "curve_comparison.csv", comparisons)
    write_csv(directory / "tables" / "diagnostics.csv", [{"source": c["source"], **c["diagnostics"]} for c in cases])
    print(f"analyzed {len(cases)} trajectories in {directory}")
