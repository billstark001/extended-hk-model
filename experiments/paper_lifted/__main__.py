"""HK + Random commitment: validate, calibrate, freeze; formal execution is explicit."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "pilot", "freeze", "run", "resume", "status", "analyze", "plot"))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--pilot-dir", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-hours", type=float, default=7)
    p.add_argument(
        "--phase",
        choices=("run", "analyze", "plot"),
        action="append",
        help="smoke/pilot/run/resume phases; repeat to select several (default: all three)",
    )
    p.add_argument("--backend", choices=("purego", "accelerate", "openblas"), default="accelerate")
    siblings = Path(__file__).resolve().parents[3]
    p.add_argument("--micro-root", type=Path, default=siblings / "social-media-models")
    p.add_argument("--meso-root", type=Path, default=siblings / "social-media-mesoscopic-models")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
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
