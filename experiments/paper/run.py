"""Original-paper simulation CLI; runtime wire arguments and output layouts are unchanged."""

import argparse

from common.io.logging import init_logger
from experiments.paths import SMP_BINARY
from experiments.workspaces import workspace_dir

from .scenarios import TASKS
from .study import run_simulations


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run simulations for a given scenario.")
    parser.add_argument("scenario", choices=TASKS.names, help="The scenario to run simulations for.")
    parser.add_argument("--workspace", help="The workspace name to store simulation results.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent simulations to run.")
    parser.add_argument("--binding-batch", action="store_true", help="let this task use the SMP binding batch runner")
    args = parser.parse_args(argv)
    directory = workspace_dir(args.workspace, args.scenario)
    directory.mkdir(parents=True, exist_ok=True)
    init_logger(None, str(directory / "logfile.log"))
    print(f"Result Directory: {directory}")
    return run_simulations(
        args.scenario, directory, SMP_BINARY.resolve(), args.concurrency, binding_batch=args.binding_batch
    )


if __name__ == "__main__":
    main()
