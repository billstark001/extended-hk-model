"""Original-paper simulation CLI; runtime wire arguments and output layouts are unchanged."""

import argparse
import json
from pathlib import Path

from common.io.logging import init_logger
from experiments.paths import SMP_BINARY
from experiments.workspaces import workspace_dir

from .recommendation.design import STUDIES
from .recommender_variants.design import STUDY as IMPLEMENTATIONS
from .scenarios import TASKS
from .study import run_simulations


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run simulations for a given scenario.")
    parser.add_argument("scenario", choices=(*TASKS.names, *STUDIES, IMPLEMENTATIONS, "recommendation"), help="The scenario to run simulations for.")
    parser.add_argument("--workspace", help="The workspace name to store simulation results.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent simulations to run.")
    parser.add_argument("--binding-batch", action="store_true", help="let this task use the SMP binding batch runner")
    parser.add_argument("--smp-binary", type=Path, help="Persistent smp executable.")
    parser.add_argument("--stage-evidence", type=Path, help="Completed staged dataset for the timing follow-up.")
    parser.add_argument("--plan-only", action="store_true", help="Print a recommendation study design without running.")
    args = parser.parse_args(argv)
    if args.scenario == IMPLEMENTATIONS:
        from .recommender_variants.study import make_plan, run_study

        if args.binding_batch:
            parser.error("implementation comparisons use persistent smp jobs")
        if args.stage_evidence:
            parser.error("implementation comparisons do not use staged evidence")
        if not 1 <= args.concurrency <= 6:
            parser.error("recommendation concurrency must be between 1 and 6")
        if args.plan_only:
            print(json.dumps(make_plan(), indent=2))
            return
        directory = workspace_dir(args.workspace, args.scenario) / args.scenario
        return run_study(directory, args.smp_binary or SMP_BINARY.resolve(), args.concurrency)
    if args.scenario in (*STUDIES, "recommendation"):
        from .recommendation.study import make_plan, run_campaign, run_study

        if args.binding_batch:
            parser.error("the recommendation studies use persistent smp jobs")
        if not 1 <= args.concurrency <= 6:
            parser.error("recommendation concurrency must be between 1 and 6")
        if args.plan_only:
            plans = [make_plan(study) for study in STUDIES] if args.scenario == "recommendation" else make_plan(args.scenario)
            print(json.dumps(plans, indent=2))
            return
        if args.scenario == "recommendation":
            if args.stage_evidence:
                parser.error("the campaign uses its own staged evidence")
            return run_campaign(workspace_dir(args.workspace), args.smp_binary or SMP_BINARY.resolve(), args.concurrency)
        directory = workspace_dir(args.workspace, args.scenario) / args.scenario
        return run_study(
            args.scenario,
            directory,
            args.smp_binary or SMP_BINARY.resolve(),
            args.concurrency,
            stage_evidence=args.stage_evidence,
        )
    if args.plan_only or args.stage_evidence:
        parser.error("--plan-only and --stage-evidence require a recommendation study")

    directory = workspace_dir(args.workspace, args.scenario)
    directory.mkdir(parents=True, exist_ok=True)
    init_logger(None, str(directory / "logfile.log"))
    print(f"Result Directory: {directory}")
    return run_simulations(
        args.scenario, directory, args.smp_binary or SMP_BINARY.resolve(), args.concurrency, binding_batch=args.binding_batch
    )


if __name__ == "__main__":
    main()
