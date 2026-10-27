"""HK + Random commitment: validate, calibrate, freeze; formal execution is explicit."""

import argparse
import json
import math
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "pilot", "freeze", "run", "resume", "status", "analyze", "plot"))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--pilot-dir", type=Path)
    p.add_argument(
        "--workers", type=int, help="execution concurrency; defaults to pilot reference, never a plan constraint"
    )
    p.add_argument("--budget-hours", type=float, default=7)
    p.add_argument(
        "--wall-hours",
        type=float,
        help="explicit total cumulative execution limit; resume can increase it without resetting used time",
    )
    p.add_argument("--schedule", choices=("balanced", "stages"), default="balanced")
    p.add_argument(
        "--core-only",
        action="store_true",
        help="complete frozen core tasks before starting optional adaptive follow-ups",
    )
    p.add_argument("--watch", action="store_true", help="status: read-only global progress monitor")
    p.add_argument("--progress", action="store_true", help="status: one lightweight global progress/ETA snapshot")
    p.add_argument("--interval", type=float, default=15, help="global progress refresh seconds")
    p.add_argument(
        "--phase",
        choices=("run", "analyze", "plot"),
        action="append",
        help="smoke/pilot/run/resume phases; repeat to select several (default: all three)",
    )
    p.add_argument("--backend", choices=("purego", "accelerate", "openblas"))
    siblings = Path(__file__).resolve().parents[3]
    p.add_argument("--micro-root", type=Path, default=siblings / "social-media-models")
    p.add_argument("--meso-root", type=Path, default=siblings / "social-media-mesoscopic-models")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        p.error("--interval must be positive")
    if args.workers is not None and not 1 <= args.workers <= 8:
        p.error("--workers must be 1..8")
    if (args.watch or args.progress) and args.action != "status":
        p.error("--watch/--progress apply only to status")
    if args.wall_hours is not None and args.action not in ("run", "resume", "smoke", "pilot"):
        p.error("--wall-hours applies to execution only")
    if args.wall_hours is not None and (not math.isfinite(args.wall_hours) or args.wall_hours <= 0):
        p.error("--wall-hours must be finite and positive")
    if args.action == "status" and (args.watch or args.progress):
        from .progress import inspect_progress

        result = inspect_progress(args.work_dir, watch=args.watch, interval=args.interval)
        if result is not None:
            print(json.dumps(result, indent=2))
        return 0
    if args.phase and args.action not in ("smoke", "pilot", "run", "resume"):
        p.error("--phase applies to smoke/pilot/run/resume; analyze and plot are independent actions")
    if args.action in ("smoke", "pilot", "run", "resume"):
        from .run import launch

        return launch(args)
    if args.action == "freeze":
        if args.pilot_dir is None:
            p.error("freeze requires --pilot-dir")
        from .evidence.budget import freeze

        print(
            json.dumps(freeze(args.pilot_dir, args.work_dir, workers=args.workers, hours=args.budget_hours), indent=2)
        )
        return 0
    if args.action == "plot":
        from .evidence.figures import plot

        plot(args.work_dir / "analysis")
        return 0
    from .evidence.analyze import analyze, status

    print(json.dumps(status(args.work_dir) if args.action == "status" else analyze(args.work_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
