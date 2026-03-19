import os
import argparse

from utils.file import init_logger
import works.config as cfg
from smp_bindings.simulation import run_simulations

scenarios = {
    "epsilon": cfg.all_scenarios_eps,
    "gradation": cfg.all_scenarios_grad,
    "mech": cfg.all_scenarios_mech,
    "replicate": cfg.all_scenarios_rep,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run simulations for a given scenario."
    )
    parser.add_argument(
        "scenario",
        choices=scenarios.keys(),
        help="The scenario to run simulations for.",
    )
    parser.add_argument(
        "--workspace", help="The workspace name to store simulation results."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent simulations to run.",
    )

    args = parser.parse_args()

    if not args.scenario:
        print("Please specify a scenario to run simulations for.")
        exit(1)
    if args.scenario not in scenarios:
        print(
            f'Invalid scenario specified. Available scenarios: {", ".join(scenarios.keys())}'
        )
        exit(1)

    SIMULATION_RESULT_DIR = cfg.get_workspace_dir(args.workspace)

    os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
    init_logger(None, os.path.join(SIMULATION_RESULT_DIR, "logfile.log"))

    print(f"Result Directory: {SIMULATION_RESULT_DIR}")

    run_simulations(
        cfg.SMP_BINARY_PATH,
        SIMULATION_RESULT_DIR,
        scenarios[args.scenario],
        max_concurrent=args.concurrency,
        show_position=True,
    )
