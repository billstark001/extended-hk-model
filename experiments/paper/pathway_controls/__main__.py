"""CLI for a frozen process-control dataset and its network inputs."""

import argparse
import json
from pathlib import Path

from experiments.paths import SMP_BINARY

from .design import CASES, make_plan
from .study import analyze, freeze, plot_snapshots, run_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "freeze", "run", "analyze", "plot"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--smp-binary", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--follow-count", type=int)
    parser.add_argument("--snapshot-steps", nargs="+", type=int)
    parser.add_argument("--seed-namespace")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--endpoint-complement", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    options = {
        key: value
        for key, value in dict(
            cases=args.cases,
            replicates=args.replicates,
            horizon=args.horizon,
            node_count=args.nodes,
            follow_count=args.follow_count,
            snapshot_steps=args.snapshot_steps,
            seed_namespace=args.seed_namespace,
        ).items()
        if value is not None
    }
    if args.action == "plan":
        print(json.dumps(make_plan(**options), indent=2))
        return
    if args.directory is None:
        parser.error("--directory is required except for plan")
    if args.action == "analyze":
        return analyze(args.directory, endpoint_complement=args.endpoint_complement)
    if args.action == "plot":
        return plot_snapshots(args.directory)
    binary = args.smp_binary or SMP_BINARY.resolve()
    if args.action == "freeze":
        return freeze(args.directory, binary, **options)
    return run_study(args.directory, binary, args.concurrency, **options)


if __name__ == "__main__":
    main()
