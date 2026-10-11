"""Generate resumable scan shards or analyze saved curves and render their fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_theory.energy_scan.expanded import config


def parse_arguments() -> argparse.Namespace:
    """Parse shard execution, input/output and analysis options from sys.argv; do not execute a solver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, help="read-only expanded scan directory")
    parser.add_argument("--output", type=Path, help="generated scan directory")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="fit/analyze and save tables only")
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument(
        "--chunk-size", type=int, default=0, help="cases per durable checkpoint; 0 uses the worker count"
    )
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--simulate-only", action="store_true", help="run/resume simulation shards without aggregate analysis"
    )
    parser.add_argument(
        "--max-new-shards", type=int, default=0, help="stop cleanly after N newly completed shards; 0 means all"
    )
    parser.add_argument("--status", action="store_true", help="show durable progress and exit")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate/resume or analyze expanded shards, fitting models and optionally rendering figures."""
    from experiments.paper_theory.energy_scan.expanded.analyze.curves import parameterize
    from experiments.paper_theory.energy_scan.expanded.analyze.report import build_summary
    from experiments.paper_theory.energy_scan.expanded.analyze.terminal import terminal_state_parameters
    from experiments.paper_theory.energy_scan.expanded.fit.edge import fit_edge_timescales
    from experiments.paper_theory.energy_scan.expanded.fit.node import low_parameter_fits
    from experiments.paper_theory.energy_scan.expanded.plot.design import plot_design, plot_relaxation_times
    from experiments.paper_theory.energy_scan.expanded.plot.parameters import plot_edge_timescale_parameters
    from experiments.paper_theory.energy_scan.expanded.plot.scaling import plot_alpha_collapse, plot_model_errors
    from experiments.paper_theory.energy_scan.expanded.plot.summary import plot_energy_two_clock_summary
    from experiments.paper_theory.energy_scan.expanded.plot.terminal import plot_terminal_energies_and_wells
    from experiments.paper_theory.energy_scan.expanded.run import load_or_run_shards, print_status
    from experiments.paper_theory.energy_scan.expanded.storage import save_csv, save_descriptor_grid

    arguments = parse_arguments()
    config.HERE = arguments.output.expanduser().resolve() if arguments.output else config.HERE
    config.INPUT = arguments.input_dir.expanduser().resolve() if arguments.input_dir else config.INPUT
    config.RAW, config.CHECKPOINTS, config.FIGURES = (
        config.HERE / "raw_shards",
        config.HERE / "case_checkpoints",
        config.HERE / "figures",
    )
    config.KINETIC = arguments.binary.expanduser().resolve() if arguments.binary else config.KINETIC
    if arguments.analysis_only and arguments.rerun:
        raise ValueError("--analysis-only cannot be combined with --rerun")
    if not arguments.analysis_only and (not arguments.status) and (not config.KINETIC.is_file()):
        raise FileNotFoundError(f"Go kinetic solver not found: {config.KINETIC}")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    chunk_size = arguments.chunk_size or arguments.workers
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if arguments.max_new_shards < 0:
        raise ValueError("max-new-shards must be nonnegative")
    if arguments.status:
        print_status(config.INPUT if arguments.input_dir else None)
        return
    config.HERE.mkdir(parents=True, exist_ok=True)
    shards, complete = load_or_run_shards(
        workers=arguments.workers,
        chunk_size=chunk_size,
        analysis_only=arguments.analysis_only,
        rerun=arguments.rerun,
        max_new_shards=arguments.max_new_shards,
    )
    if arguments.simulate_only or not complete:
        print_status()
        return
    rows = parameterize(shards)
    fits = low_parameter_fits(rows)
    edge_timescale_fits = fit_edge_timescales(shards)
    terminal_parameters = terminal_state_parameters(shards, rows)
    save_csv(config.HERE / "curve_parameters.csv", rows)
    save_csv(config.HERE / "low_parameter_fits.csv", fits)
    save_csv(config.HERE / "edge_timescale_fits.csv", edge_timescale_fits)
    save_csv(config.HERE / "terminal_state_parameters.csv", terminal_parameters)
    save_descriptor_grid(rows)
    if not arguments.no_plots:
        plot_design()
        plot_relaxation_times(rows)
        plot_edge_timescale_parameters(edge_timescale_fits, fits)
        plot_energy_two_clock_summary(rows, edge_timescale_fits, fits)
        plot_terminal_energies_and_wells(terminal_parameters)
        plot_alpha_collapse(shards)
        plot_model_errors(fits)
    summary = build_summary(rows, fits)
    summary["source_protocols"] = {
        f"{d}_alpha_{a:.12g}": json.loads(str(data["protocol"])) for (d, a), data in shards.items()
    }
    (config.HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary["quality"], indent=2), flush=True)
