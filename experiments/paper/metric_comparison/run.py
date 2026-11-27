"""Collect saved metrics, align them and produce a report in one resumable run."""

import argparse
import json
from pathlib import Path

from common.experiments.lifecycle import interruptible, session_lock
from common.io.json import atomic_json, sha

from . import core, raw, report

RETRYABLE_SOURCE_CHANGE = "raw source files changed during legacy metric calculation"


def _collection_summary(result):
    fields = ("status", "planned_paths", "selected_paths", "completed_paths", "pending_paths", "failed_paths", "points")
    return {
        **{name: result[name] for name in fields if name in result},
        "failures": [item for item in result.get("paths", ()) if item.get("status") == "failed"],
    }


def _complete_collection(result):
    planned, completed = result.get("planned_paths"), result.get("completed_paths")
    return (
        result.get("status") == "complete" and type(planned) is int and type(completed) is int
        and planned == completed and result.get("pending_paths", 0) == 0 and result.get("failed_paths", 0) == 0
    )


def _retryable(result):
    failures = [item for item in result.get("paths", ()) if item.get("status") == "failed"]
    return (
        result.get("status") == "failed" and result.get("pending_paths", 0) == 0
        and bool(failures) and result.get("failed_paths") == len(failures)
        and all(item.get("error") == RETRYABLE_SOURCE_CHANGE for item in failures)
    )


def run(preparation, output, *, collection=None, workers=6, attempts=2, make_figures=True, endpoint_complement=True):
    """Finish an offline comparison, stopping explicitly when any data remain missing.

    Collection owns authenticated per-path recovery. Only source-change failures
    may retry; alignment and reports always require new destination directories.
    """
    if type(workers) is not int or workers < 1 or type(attempts) is not int or attempts < 1:
        raise ValueError("workers and attempts must be positive integers")
    preparation, output = Path(preparation).expanduser().resolve(), Path(output).expanduser().resolve()
    collection = Path(collection).expanduser().resolve() if collection is not None else output / "collection"
    alignment, destination = output / "alignment", output / "report"
    if preparation == output or preparation in output.parents or output in preparation.parents:
        raise ValueError("pipeline output must be separate from the preparation")
    if collection == output or any(
        collection == target or collection in target.parents or target in collection.parents
        for target in (alignment, destination)
    ):
        raise ValueError("collection must be separate from alignment and report destinations")
    status = {
        "schema": 1, "state": "running", "phase": "preflight", "attempt": 0,
        "workers": workers, "attempts": attempts, "make_figures": make_figures,
        "endpoint_complement": endpoint_complement,
        "runner_sha256": sha(Path(__file__)),
        "paths": {"preparation": str(preparation), "output": str(output), "collection": str(collection),
                  "alignment": str(alignment), "report": str(destination),
                  "collection_progress": str(collection / "progress.json")},
    }

    def publish(phase, state, **details):
        status.update(phase=phase, state=state, **details)
        atomic_json(output / "run_status.json", status)

    with session_lock(output / "runtime"), interruptible():
        try:
            publish("preflight", "running")
            for target in (alignment, destination):
                if target.exists():
                    raise ValueError(f"pipeline destination already exists; choose a new output directory: {target}")
            for attempt in range(1, attempts + 1):
                publish("collect", "running", attempt=attempt)
                collected = raw.collect_raw(preparation, collection, workers=workers, endpoint_complement=endpoint_complement)
                status["collection"] = _collection_summary(collected)
                if _complete_collection(collected):
                    break
                if attempt < attempts and _retryable(collected):
                    publish("collect", "retrying", reason=RETRYABLE_SOURCE_CHANGE)
                    continue
                state = "failed" if collected.get("status") == "failed" else "pending"
                publish("collect", state, reason="collection is incomplete; alignment and report were not started")
                return status
            status.pop("reason", None)
            publish("alignment", "running")
            aligned = core.compare(preparation, collection, alignment)
            status["alignment"] = aligned
            if aligned.get("status") != "complete":
                publish("alignment", "failed" if aligned.get("status") == "failed" else "pending",
                        reason="exact alignment is incomplete; report was not started")
                return status
            publish("report", "running")
            reported = report.report(preparation, alignment, destination, make_figures=make_figures)
            status["report"] = reported
            if reported.get("status") != "complete":
                publish("report", "failed" if reported.get("status") == "failed" else "pending",
                        reason="report did not confirm complete coverage")
                return status
            publish("report", "complete")
            return status
        except BaseException as error:
            publish(status["phase"], "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", type=Path, help="Reuse an existing collection and its committed path receipts")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--endpoint-complement", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = run(args.preparation, args.output, collection=args.collection, workers=args.workers,
                 attempts=args.attempts, make_figures=not args.no_figures, endpoint_complement=args.endpoint_complement)
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
