"""CLI for a frozen process-control dataset and its network inputs."""

import argparse
import json
from pathlib import Path

from experiments.paths import SMP_BINARY
from common.io.json import json_default

from .design import CASES, ControlDesign, SEED_NAMESPACE
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
    parser.add_argument("--force", action="store_true", help="Recompute the selected control metrics.")
    args = parser.parse_args(argv)
    changed_design = any(value is not None for value in (
        args.cases, args.replicates, args.horizon, args.nodes, args.follow_count,
        args.snapshot_steps, args.seed_namespace))
    design = ControlDesign(
        cases=tuple(args.cases) if args.cases is not None else CASES,
        replicates=args.replicates if args.replicates is not None else 12,
        horizon=args.horizon if args.horizon is not None else 2048,
        node_count=args.nodes if args.nodes is not None else 500,
        follow_count=args.follow_count if args.follow_count is not None else 15,
        snapshot_steps=tuple(args.snapshot_steps) if args.snapshot_steps is not None else None,
        seed_namespace=args.seed_namespace if args.seed_namespace is not None else SEED_NAMESPACE,
    )
    if args.action == "plan":
        print(json.dumps(design.plan(), indent=2, default=json_default))
        return
    if args.directory is None:
        parser.error("--directory is required except for plan")
    if args.action == "analyze":
        return analyze(args.directory, endpoint_complement=args.endpoint_complement, concurrency=args.concurrency, force=args.force)
    if args.action == "plot":
        return plot_snapshots(args.directory)
    binary = args.smp_binary or SMP_BINARY.resolve()
    if args.action == "freeze":
        return freeze(args.directory, binary, design if changed_design else None)
    return run_study(args.directory, binary, args.concurrency, design if changed_design else None)


if __name__ == "__main__":
    result = main()
    if result is not None and getattr(result, "state", None) in ("partial", "failed", "interrupted"):
        raise SystemExit(1)
