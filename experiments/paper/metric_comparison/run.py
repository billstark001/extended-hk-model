"""Collect saved metrics, align them and produce a report in one resumable run."""

import argparse
import json
from pathlib import Path
from dataclasses import replace

from ehk.domain import to_wire
from .contracts import CollectionManifest, CollectionSummary, PipelinePaths, PipelineStatus

from common.experiments.lifecycle import interruptible, session_lock
from common.io.json import atomic_json, sha

from . import core, raw, report

RETRYABLE_SOURCE_CHANGE = "raw source files changed during canonical metric calculation"


def _collection_summary(result: CollectionManifest) -> CollectionSummary:
    return CollectionSummary(result.status, result.planned_paths, result.selected_paths,
        result.completed_paths, result.pending_paths, result.failed_paths, result.points,
        tuple(item for item in result.paths if item.status == "failed"))


def _complete_collection(result: CollectionManifest) -> bool:
    return (result.status == "complete" and result.planned_paths == result.completed_paths
            and result.pending_paths == 0 and result.failed_paths == 0)


def _retryable(result: CollectionManifest) -> bool:
    failures = tuple(item for item in result.paths if item.status == "failed")
    return (result.status == "failed" and result.pending_paths == 0 and bool(failures)
            and result.failed_paths == len(failures)
            and all(item.error == RETRYABLE_SOURCE_CHANGE for item in failures))


def run(preparation, output, *, collection=None, workers=6, attempts=2, make_figures=True, endpoint_complement=True):
    """Finish an offline comparison, stopping explicitly when any data remain missing.

    Collection owns authenticated per-path recovery. Only source-change failures
    may retry; authenticated alignment and report artifacts are resumed.
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
    status = PipelineStatus(schema=1, state="running", phase="preflight", attempt=0,
        workers=workers, attempts=attempts, make_figures=make_figures,
        endpoint_complement=endpoint_complement, runner_sha256=sha(Path(__file__)),
        paths=PipelinePaths(str(preparation), str(output), str(collection), str(alignment),
                            str(destination), str(collection / "progress.json")))

    def publish(phase, state, **details):
        nonlocal status
        status = replace(status, phase=phase, state=state, **details)
        atomic_json(output / "run_status.json", status)

    with session_lock(output / "runtime"), interruptible():
        try:
            publish("preflight", "running")
            for target,marker in ((alignment,"comparison_inputs.json"),(destination,"report_inputs.json")):
                if target.exists() and not (target/marker).is_file():
                    raise ValueError(f"pipeline destination already exists without matching inputs: {target}")
            for attempt in range(1, attempts + 1):
                publish("collect", "running", attempt=attempt)
                collected = raw.collect_raw(preparation, collection, workers=workers, endpoint_complement=endpoint_complement)
                status = replace(status, collection=_collection_summary(collected))
                if _complete_collection(collected):
                    break
                if attempt < attempts and _retryable(collected):
                    publish("collect", "retrying", reason=RETRYABLE_SOURCE_CHANGE)
                    continue
                state = "failed" if collected.status == "failed" else "pending"
                publish("collect", state, reason="collection is incomplete; alignment and report were not started")
                return status
            status = replace(status, reason=None)
            publish("alignment", "running")
            aligned = core.compare(preparation, collection, alignment)
            status = replace(status, alignment=aligned)
            if aligned.status != "complete":
                publish("alignment", "failed" if aligned.status == "failed" else "pending",
                        reason="exact alignment is incomplete; report was not started")
                return status
            publish("report", "running")
            reported = report.report(preparation, alignment, destination, make_figures=make_figures)
            status = replace(status, report=reported)
            if reported.status != "complete":
                publish("report", "failed" if reported.status == "failed" else "pending",
                        reason="report did not confirm complete coverage")
                return status
            publish("report", "complete")
            return status
        except BaseException as error:
            publish(status.phase, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
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
    print(json.dumps(to_wire(result), indent=2))
    return 0 if result.state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
