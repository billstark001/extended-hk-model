"""Compute parameter-derived clocks or redraw saved clock tables without simulations."""

import argparse
import json
from pathlib import Path

from experiments.paper_theory.release_clocks.artifact_paths import ensure_output_dirs

from ..artifact_paths import CLOCK_INPUT_DATA
from . import config
from .analyze import generate
from .data import read_saved_table
from .plot import plots


def main():
    """Compute clocks from saved fits, or redraw existing typed tables without solver caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only", action="store_true", help="Redraw saved result tables; does not load the original solver cache."
    )
    parser.add_argument("--output-dir", type=Path, help="Optional figure destination, useful for checking a redraw.")
    args = parser.parse_args()
    ensure_output_dirs()
    if args.output_dir:
        config.CLOCK_FIGURES = args.output_dir.resolve()
        config.CLOCK_FIGURES.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        rows = read_saved_table(CLOCK_INPUT_DATA / "parameter_metrics.csv")
        qs = read_saved_table(CLOCK_INPUT_DATA / "release_times.csv")
    else:
        rows, qs = generate()
    plots(rows, qs)
    print(
        json.dumps(
            json.loads(((CLOCK_INPUT_DATA if args.plots_only else config.OUT) / "summary.json").read_text())[
                "curve_errors"
            ],
            indent=2,
        )
    )
