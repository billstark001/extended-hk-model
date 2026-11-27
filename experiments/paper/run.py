"""Original-paper simulation CLI; runtime wire arguments and output layouts are unchanged."""

import argparse
import json
from pathlib import Path

from common.io.logging import init_logger
from common.io.json import json_default
from experiments.paths import SMP_BINARY
from experiments.workspaces import workspace_dir

from .recommendation.design import STUDIES
from .recommender_variants.design import STUDY as IMPLEMENTATIONS
from .scenarios import SEED_NAMESPACE, TASKS
from .study import freeze_study, run_simulations
from .study import make_plan as original_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run simulations for a given scenario.")
    parser.add_argument(
        "scenario",
        choices=(*TASKS.names, *STUDIES, IMPLEMENTATIONS, "recommendation"),
        help="The scenario to run simulations for.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--workspace", help="The workspace name to store simulation results.")
    destination.add_argument(
        "--run-dir", type=Path, help="New original-paper output directory, without editing sim_ws.json."
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent simulations to run.")
    parser.add_argument("--smp-binary", type=Path, help="Persistent smp executable.")
    parser.add_argument("--stage-evidence", type=Path, help="Completed staged dataset for the timing follow-up.")
    parser.add_argument("--plan-only", action="store_true", help="Print the study design without running.")
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Freeze an original-paper run and executable provenance without running.",
    )
    parser.add_argument(
        "--seed-namespace", default=SEED_NAMESPACE, help="Stable seed namespace for original-paper recalculations."
    )
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.scenario not in TASKS.names:
        parser.error("--run-dir is supported for original-paper scenarios")
    if args.scenario == IMPLEMENTATIONS:
        from .recommender_variants.study import make_plan, run_study

        if args.freeze_only:
            parser.error("--freeze-only is for original-paper scenarios")
        if args.stage_evidence:
            parser.error("implementation comparisons do not use staged evidence")
        if not 1 <= args.concurrency <= 6:
            parser.error("recommendation concurrency must be between 1 and 6")
        if args.plan_only:
            print(json.dumps(make_plan(), indent=2, default=json_default))
            return
        directory = workspace_dir(args.workspace, args.scenario) / args.scenario
        return run_study(directory, args.smp_binary or SMP_BINARY.resolve(), args.concurrency)
    if args.scenario in (*STUDIES, "recommendation"):
        from .recommendation.study import make_plan, run_campaign, run_study

        if args.freeze_only:
            parser.error("--freeze-only is for original-paper scenarios")
        if not 1 <= args.concurrency <= 6:
            parser.error("recommendation concurrency must be between 1 and 6")
        if args.plan_only:
            plans = (
                [make_plan(study) for study in STUDIES]
                if args.scenario == "recommendation"
                else make_plan(args.scenario)
            )
            print(json.dumps(plans, indent=2, default=json_default))
            return
        if args.scenario == "recommendation":
            if args.stage_evidence:
                parser.error("the campaign uses its own staged evidence")
            return run_campaign(
                workspace_dir(args.workspace), args.smp_binary or SMP_BINARY.resolve(), args.concurrency
            )
        directory = workspace_dir(args.workspace, args.scenario) / args.scenario
        return run_study(
            args.scenario,
            directory,
            args.smp_binary or SMP_BINARY.resolve(),
            args.concurrency,
            stage_evidence=args.stage_evidence,
        )
    if args.stage_evidence:
        parser.error("--stage-evidence requires a recommendation study")
    if args.plan_only:
        print(json.dumps(original_plan(args.scenario, seed_namespace=args.seed_namespace), indent=2, default=json_default))
        return

    directory = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else workspace_dir(args.workspace, args.scenario)
    )
    directory.mkdir(parents=True, exist_ok=True)
    if args.freeze_only:
        return freeze_study(
            args.scenario, directory, args.smp_binary or SMP_BINARY.resolve(), seed_namespace=args.seed_namespace
        )
    init_logger(None, str(directory / "logfile.log"))
    print(f"Result Directory: {directory}")
    return run_simulations(
        args.scenario,
        directory,
        args.smp_binary or SMP_BINARY.resolve(),
        args.concurrency,
        seed_namespace=args.seed_namespace,
    )


if __name__ == "__main__":
    result = main()
    if result is not None and getattr(result, "state", None) in ("partial", "failed", "interrupted"):
        raise SystemExit(1)
