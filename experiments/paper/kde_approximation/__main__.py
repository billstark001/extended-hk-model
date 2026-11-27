"""Run the isolated binned-KDE experiment against a completed exact baseline."""

import argparse
import json
from pathlib import Path

from common.experiments.lifecycle import interruptible, session_lock
from common.io.json import atomic_json, sha

from . import collector, report


def run(preparation, baseline, output, *, bins=collector.DEFAULT_BINS, workers=4, max_tasks=32, full=False,
        benchmark_states=64, repeats=3, make_figures=True, endpoint_complement=True):
    preparation, baseline, output = (Path(path).expanduser().resolve() for path in (preparation, baseline, output))
    if any(output == source or output in source.parents or source in output.parents for source in (preparation, baseline)):
        raise ValueError("experiment output must be separate from frozen evidence")
    status = {"schema": 1, "state": "running", "phase": "collect", "runner_sha256": sha(Path(__file__)),
              "endpoint_complement": endpoint_complement,
              "paths": {"preparation": str(preparation), "baseline": str(baseline),
                        "collection": str(output / "collection"), "report": str(output / "report")}}
    with session_lock(output / "runtime"), interruptible():
        if (output / "report").exists():
            raise ValueError("experiment report already exists; choose a new output directory")
        try:
            atomic_json(output / "run_status.json", status)
            collected = collector.collect(preparation, baseline, output / "collection", bins=bins, workers=workers,
                                          max_tasks=max_tasks, full=full, benchmark_states=benchmark_states, repeats=repeats,
                                          endpoint_complement=endpoint_complement)
            status["collection"] = {key: value for key, value in collected.items() if key not in ("paths", "files")}
            if collected["status"] != "complete":
                status["state"] = collected["status"]
                atomic_json(output / "run_status.json", status)
                return status
            status["phase"] = "report"
            atomic_json(output / "run_status.json", status)
            result = report.report(preparation, baseline / "report", output / "collection", output / "report",
                                   make_figures=make_figures)
            status["report"] = result
            status["state"] = result["status"]
            atomic_json(output / "run_status.json", status)
            return status
        except BaseException as error:
            status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            atomic_json(output / "run_status.json", status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Completed exact metric-comparison pipeline root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=32)
    parser.add_argument("--full", action="store_true", help="Freeze all initial-state groups in a separate experiment")
    parser.add_argument("--bins", type=int, nargs="+", default=collector.DEFAULT_BINS)
    parser.add_argument("--benchmark-states", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--endpoint-complement", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = run(args.preparation, args.baseline, args.output, bins=args.bins, workers=args.workers,
                 max_tasks=args.max_tasks, full=args.full, benchmark_states=args.benchmark_states,
                 repeats=args.repeats, make_figures=not args.no_figures, endpoint_complement=args.endpoint_complement)
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
