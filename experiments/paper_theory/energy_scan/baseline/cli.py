"""Generate or analyze the baseline scan and export its tables, interpolants and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper_theory.energy_scan.baseline import config


def main() -> None:
    """Generate or analyze baseline energy curves, exporting descriptors, interpolants and figures."""
    from experiments.paper_theory.energy_scan.baseline.analyze.curves import parameterize
    from experiments.paper_theory.energy_scan.baseline.analyze.report import build_summary
    from experiments.paper_theory.energy_scan.baseline.design import protocol
    from experiments.paper_theory.energy_scan.baseline.plot.curves import (
        plot_descriptors,
        plot_energy_curves,
        plot_final_potentials,
        plot_first_step,
    )
    from experiments.paper_theory.energy_scan.baseline.run import load_or_run
    from experiments.paper_theory.energy_scan.baseline.storage import save_descriptor_csv, save_interpolants

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="read-only cached baseline directory")
    parser.add_argument("--output", type=Path, help="generated results directory")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    config.HERE = args.output.expanduser().resolve() if args.output else config.HERE
    config.INPUT = args.input_dir.expanduser().resolve() if args.input_dir else config.HERE
    config.KINETIC = args.binary.expanduser().resolve() if args.binary else config.KINETIC
    if not args.analysis_only and (not config.KINETIC.is_file()):
        raise FileNotFoundError(f"Go kinetic solver not found: {config.KINETIC}")
    if args.analysis_only:
        with np.load(config.INPUT / "energy_curve_parameterization.npz", allow_pickle=False) as archive:
            recorded = json.loads(str(archive["protocol"]))
            config.ALPHA = float(archive["alpha"])
            config.EPSILONS, config.RATIOS = (archive["epsilon"].copy(), archive["q_over_alpha"].copy())
            config.PROGRESS_LEVELS = archive["progress_levels"].copy()
            config.GRID_SIZE = int(recorded["grid_size"])
            config.MAX_STEPS, config.RECORD_EVERY = (int(recorded["maximum_steps"]), int(recorded["record_every"]))
    else:
        recorded = protocol()
    config.HERE.mkdir(parents=True, exist_ok=True)
    data, audit = load_or_run(analysis_only=args.analysis_only)
    if args.simulate_only:
        return
    rows, quantiles = parameterize(data)
    save_descriptor_csv(rows)
    save_interpolants(rows, quantiles)
    summary = build_summary(data, rows, audit, recorded)
    plot_energy_curves(data)
    plot_descriptors(rows)
    plot_first_step(audit)
    plot_final_potentials(data)
    print(json.dumps(summary["scan"], indent=2), flush=True)
