"""Generate or analyze channel-scaling probes and write diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_theory.energy_scan.scaling import config


def main() -> None:
    """Generate or analyze channel probes and dynamic-collapse caches, then write summaries and figures."""
    from experiments.paper_theory.energy_scan.scaling.analyze import build_summary
    from experiments.paper_theory.energy_scan.scaling.plot import (
        plot_channel_tests,
        plot_dynamic_collapse,
        plot_epsilon_derivatives,
        plot_field_collapse,
    )
    from experiments.paper_theory.energy_scan.scaling.run import load_or_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    config.HERE = args.output.expanduser().resolve() if args.output else config.HERE
    config.INPUT = args.input_dir.expanduser().resolve() if args.input_dir else config.HERE
    config.KINETIC = args.binary.expanduser().resolve() if args.binary else config.KINETIC
    if not args.analysis_only and (not config.KINETIC.is_file()):
        raise FileNotFoundError(f"Go solver not found: {config.KINETIC}")
    config.HERE.mkdir(parents=True, exist_ok=True)
    data = load_or_run(analysis_only=args.analysis_only)
    summary = build_summary(data)
    plot_channel_tests(data)
    plot_dynamic_collapse(data)
    plot_field_collapse(data)
    plot_epsilon_derivatives(data)
    print(json.dumps(summary["frozen_channel_tests"], indent=2), flush=True)
