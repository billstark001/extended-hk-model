"""Typed, recoverable collection of canonical indices from saved histories."""

from __future__ import annotations

import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.experiments.progress import Progress
from common.io.json import atomic_json, fingerprint, sha
from common.runtime.process import run_tasks
from common.storage.documents import DocumentWrite, DocumentWriter, encode_document
from ehk.domain import Rng, RecommenderParams, ScenarioMetadata, parse_binding, parse_dataclass, parse_metadata, rows_frame
from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.protocol import metric_protocol
from ehk.metrics.state import MeasurementRecord
from experiments.paper.recommendation_shared.storage import read_table, write_table

from .contracts import (
    CacheCounts, CollectionConfiguration, CollectionJob, CollectionManifest, CollectionOutcome,
    CollectionProgress, CollectionReceipt, DistanceParameters, EvaluationRecord, MetricPoint,
    PathJob, PathMeasurement, PathMetric, PathProvenance, PathStatus, PreparedPath, PreparedPoint,
    SourceConfiguration, StatusCounts, table_records,
)
from .core import IW_SAMPLING_METHOD
from .receipts import receipt

PATH_KEYS = ("profile", "task_id", "stage", "arm")
SOURCE_KEYS = ("plan_fingerprint", "binary_sha256", "raw_record")
POINT_COLUMNS = (*PATH_KEYS, "step", *SOURCE_KEYS, "I_p", "I_s", "I_h")
PATH_COLUMNS = (*PATH_KEYS, *SOURCE_KEYS, "I_w", "sampling_method", "prefix_start", "trajectory_end", "endpoint_complement")


def _source_configuration() -> SourceConfiguration:
    from smp_bindings import RawSimulationRecord
    import ehk.domain
    import smp_bindings.domain
    import smp_bindings.model_state
    import smp_bindings.events_db
    import smp_bindings.trajectory
    from common.numerics import adaptive_discrete_sampling, area_under_curve
    from common.numerics.kde import get_kde_pdf
    from ehk.metrics.density import FittedDensity
    from ehk.metrics.homophily import mean_followee_edge_concordance
    from ehk.metrics.polarization import DistanceCalculator
    from ehk.metrics.state import MeasurementState
    from ehk.metrics.trajectory import sample_indices
    from ehk.micro import observables

    sources = {Path(inspect.getfile(item)).resolve() for item in (
        RawSimulationRecord, mean_followee_edge_concordance, DistanceCalculator,
        observables.calc_distance, adaptive_discrete_sampling, area_under_curve, get_kde_pdf,
        MeasurementEvaluator, MeasurementState, sample_indices, FittedDensity,
    )}
    sources.update(Path(module.__file__).resolve() for module in (
        ehk.domain, smp_bindings.domain, smp_bindings.model_state, smp_bindings.events_db, smp_bindings.trajectory,
    ))
    calculator = observables.distance_collector
    return SourceConfiguration(
        sha(Path(__file__)), {str(path): sha(path) for path in sorted(sources)},
        {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        DistanceParameters(calculator.min_bandwidth, calculator.sample_count, calculator.max_kde_samples,
                           calculator.t_err, calculator.k),
        IW_SAMPLING_METHOD, "canonical linear-binned Gaussian KDE; 4081 internal bins",
        "saved_prepared_steps; absolute_step_rng_seed",
        "positive_out_degree_users_equal_weight; existing_normalize_homophily",
        "full_saved_history_from_step_zero",
    )


def _graph_snapshots(directory):
    pattern = re.compile(r"graph-(\d+)\.msgpack(?:\.lz4)?$")
    return [(int(match.group(1)), path) for path in sorted(directory.glob("graph-*"))
            if path.is_file() and (match := pattern.fullmatch(path.name))]


def _required_files(directory):
    required = ("metadata.json", "trajectory.json", "events.db")
    missing = [name for name in required if not (directory / name).is_file()]
    if not any(step == 0 for step, _ in _graph_snapshots(directory)):
        missing.append("graph-0.msgpack[.lz4]")
    if not any(path.is_file() for path in directory.glob("finished*")):
        missing.append("finished*")
    return tuple(missing)


@dataclass(frozen=True, slots=True)
class RngIdentity:
    algorithm: str
    seed1: int
    seed2: int


def _rng_identity(value: Rng | None) -> RngIdentity | None:
    if value is None:
        return None
    def integer(seed):
        text = str(seed)
        return int(text, 16 if text.lower().startswith("0x") else 10)
    return RngIdentity(value.Algorithm, integer(value.Seed1), integer(value.Seed2))


def _validate_raw(directory: Path, path: PreparedPath, points: tuple[PreparedPoint, ...]) -> ScenarioMetadata:
    metadata = parse_metadata(json.loads((directory / "metadata.json").read_text()))
    if metadata.UniqueName != directory.name:
        raise ValueError("raw metadata name differs from its directory")
    if float(metadata.HKParams.Tolerance) != path.epsilon:
        raise ValueError("raw confidence radius differs from the prepared task")
    expected = parse_metadata(json.loads(path.metadata_json))
    if metadata.HKParams != expected.HKParams or metadata.NodeCount != expected.NodeCount:
        raise ValueError("raw population or dynamics differ from the prepared task")
    if _rng_identity(metadata.RNG) != _rng_identity(expected.RNG):
        raise ValueError("raw initialization seed differs from the prepared task")
    for name in ("PostRetainCount", "RecsysCount", "RecsysFactoryType"):
        if getattr(expected, name) is not None and getattr(metadata, name) != getattr(expected, name):
            raise ValueError(f"raw {name} differs from the prepared task")
    parameters = parse_dataclass(RecommenderParams, json.loads(path.expected_rec_params_json))
    for definition in fields(parameters):
        if definition.name.startswith("_"):
            continue
        value = getattr(parameters, definition.name)
        if value is not None and (metadata.RecSysParams is None or getattr(metadata.RecSysParams, definition.name) != value):
            raise ValueError("raw recommendation parameters differ from the prepared arm")
    dataset = directory
    for _ in Path(path.raw_record).parts:
        dataset = dataset.parent
    binding_file = dataset / "runtime" / "binding.json"
    if binding_file.is_file():
        binding = parse_binding(json.loads(binding_file.read_text()))
        actual = (binding.arm_binary_sha256.get(path.arm) if binding.arm_binary_sha256 is not None
                  else binding.binary_sha256)
        if actual != path.binary_sha256:
            raise ValueError("mounted raw dataset binary differs from the prepared arm")
    return metadata


def _raw_sources(directory):
    files = {directory / name for name in ("metadata.json", "trajectory.json", "events.db")}
    files.update(path for _, path in _graph_snapshots(directory))
    wal = directory / "events.db-wal"
    if wal.is_file() and wal.stat().st_size:
        files.add(wal)
    return {path.name: sha(path) for path in sorted(files)}


def _measure(directory: Path, path: PreparedPath, points: tuple[PreparedPoint, ...], metrics=None,
             *, endpoint_complement=True) -> PathMeasurement:
    from smp_bindings import RawSimulationRecord
    from ehk.metrics.trajectory import pathway_index, sample_indices

    metrics = metrics or MeasurementEvaluator()
    metadata = _validate_raw(directory, path, points)
    identity = path.identity
    with RawSimulationRecord(str(directory.parent), metadata) as saved:
        record = MeasurementRecord.from_saved_record(saved)
        if record.max_step != path.end_step:
            raise ValueError("raw trajectory has a different final step")
        values = []
        for point in points:
            step = point.step
            ip, isub, ih = metrics.point(record, step)
            if not np.isfinite([ip, isub, ih]).all():
                raise ValueError(f"existing metric is not finite at step {step}")
            values.append(MetricPoint(identity.profile, identity.task_id, identity.stage, identity.arm,
                                      identity.plan_fingerprint, identity.binary_sha256, identity.raw_record,
                                      step, float(ip), float(isub), float(ih)))
        def point(step):
            result = metrics.point(record, int(step))
            return result
        steps, h, p, _ = sample_indices(record.max_step, point)
        if not len(steps) or int(steps[0]) != 0 or int(steps[-1]) != record.max_step:
            raise ValueError("existing trajectory-index calculation omitted the full prefix")
        iw = float(pathway_index(p, h, endpoint_complement=endpoint_complement))
        if not np.isfinite(iw):
            raise ValueError("existing pathway index is not finite")
        trajectory = PathMetric(identity.profile, identity.task_id, identity.stage, identity.arm,
                                identity.plan_fingerprint, identity.binary_sha256, identity.raw_record,
                                iw, IW_SAMPLING_METHOD, 0, record.max_step, endpoint_complement)
    return PathMeasurement(tuple(values), trajectory,
                           EvaluationRecord(tuple(int(step) for step in steps), endpoint_complement))


_WORKER_EVALUATOR = None


def _collect_group(job: CollectionJob) -> CollectionOutcome:
    global _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is None:
        _WORKER_EVALUATOR = MeasurementEvaluator()
    cache = _WORKER_EVALUATOR
    before = cache.cache_info()
    statuses, receipts = [], []
    for entry in job.paths:
        path, observations = entry.path, entry.points
        identity = path.identity
        saved = receipt(job.output, "metric_comparison.paths", identity, CollectionReceipt)
        directory = Path(path.raw_directory)
        if path.profile in job.raw_roots:
            directory = Path(job.raw_roots[path.profile]) / path.raw_record
        missing = _required_files(directory)
        state = PathStatus.pending(identity)
        state = replace(state, raw_directory=str(directory))
        if saved is not None:
            if saved.configuration_fingerprint != job.configuration_fingerprint or saved.identity != identity:
                raise ValueError("cached canonical metrics differ from the preparation or implementation")
            if not missing and _raw_sources(directory) != saved.raw_sha256:
                raise ValueError("raw source files changed after cached canonical metrics were calculated")
            state = replace(state, status="cached")
        elif missing:
            state = replace(state, missing_files=missing)
        else:
            try:
                sources = _raw_sources(directory)
                measured = _measure(directory, path, observations, cache, endpoint_complement=job.endpoint_complement)
                if _raw_sources(directory) != sources:
                    raise ValueError("raw source files changed during canonical metric calculation")
            except (OSError, ValueError, KeyError) as error:
                state = replace(state, status="failed", error=str(error))
            else:
                receipts.append(CollectionReceipt(job.configuration_fingerprint, identity,
                                                   job.endpoint_complement, sources, measured))
                state = replace(state, status="complete")
        statuses.append(state)
    after = cache.cache_info()
    return CollectionOutcome(tuple(statuses), tuple(receipts), after.total_hits - before.total_hits,
                             after.total_misses - before.total_misses)


def collect_raw(preparation, output, *, max_paths=None, raw_roots=None, workers=1,
                endpoint_complement=True) -> CollectionManifest:
    from . import core
    from .core import authenticate_preparation

    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    manifest = authenticate_preparation(preparation)
    if max_paths is not None and (type(max_paths) is not int or max_paths < 1):
        raise ValueError("max_paths must be a positive integer")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    if output == preparation or preparation in output.parents or output in preparation.parents:
        raise ValueError("raw collection must use a separate output directory")
    roots = {profile: str(Path(root).expanduser().resolve()) for profile, root in (raw_roots or {}).items()}
    paths = read_table(preparation, "paths.csv")
    points = read_table(preparation, "points.csv.gz")
    groups = points.groupby(list(PATH_KEYS), sort=False).indices
    configuration = CollectionConfiguration(1, fingerprint(manifest), _source_configuration(), roots,
                                            endpoint_complement, metric_protocol(endpoint_complement=endpoint_complement))
    selected = paths if max_paths is None else paths.iloc[:max_paths]
    selected_paths = table_records(selected, PreparedPath)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outcomes: dict[PathProvenance, PathStatus] = {}
    cache_counts = CacheCounts()
    terminal_progress = Progress(len(selected_paths), stream=sys.stderr)
    failure = None

    def jobs():
        for path in selected_paths:
            key = (path.profile, path.task_id, path.stage, path.arm)
            observations = table_records(points.iloc[groups[key]], PreparedPoint)
            yield CollectionJob((path.profile, path.task_id), output, roots, fingerprint(configuration),
                                endpoint_complement, (PathJob(path, observations),))

    def progress(state, last=None):
        counts = StatusCounts.count(outcomes.values())
        atomic_json(output / "progress.json", CollectionProgress(
            state, workers, len(paths), len(selected_paths), counts.complete + counts.cached,
            counts, time.monotonic() - started, cache_counts, last,
        ))

    def finished(job: CollectionJob, result: CollectionOutcome):
        nonlocal cache_counts
        writer.put_many(DocumentWrite("metric_comparison.paths", fingerprint(saved.identity), saved) for saved in result.receipts)
        for state in result.paths:
            outcomes[PathProvenance(state.profile, state.task_id, state.stage, state.arm,
                                    state.plan_fingerprint, state.binary_sha256, state.raw_record)] = state
            terminal_progress.event(state.key.label, "deferred" if state.status == "pending" else state.status)
        cache_counts = CacheCounts(cache_counts.hits + result.cache_hits, cache_counts.misses + result.cache_misses)
        progress("running", job.id)

    with session_lock(output / "runtime"), interruptible(), DocumentWriter(output / "analysis.sqlite") as writer:
        freeze_manifest(output / "configuration.json", configuration)
        progress("running")
        try:
            if workers == 1:
                for job in jobs():
                    terminal_progress.event(job.paths[0].path.key.label, "running")
                    finished(job, _collect_group(job))
            else:
                run_tasks(_collect_group, jobs(), workers, executor="process", on_result=finished,
                          event=lambda job, state, **details: terminal_progress.event(job.paths[0].path.key.label, state))
        except BaseException as error:
            failure = error
        all_points, all_paths, statuses = [], [], []
        for path in selected_paths:
            identity = path.identity
            document = writer.get("metric_comparison.paths", fingerprint(identity))
            saved = None if document is None else parse_dataclass(CollectionReceipt, document)
            state = outcomes.get(identity, PathStatus.pending(identity))
            if saved is not None:
                if saved.configuration_fingerprint != fingerprint(configuration) or saved.identity != identity:
                    raise ValueError("path receipt disagrees with the frozen collection configuration")
                all_points.extend(saved.result.points)
                all_paths.append(saved.result.path)
                if state.status == "pending":
                    state = replace(state, status="complete")
            statuses.append(state)
        for name, frame in (("indices.csv.gz", rows_frame(all_points, MetricPoint)),
                            ("pathways.csv", rows_frame(all_paths, PathMetric))):
            write_table(output,name,frame,writer=writer)
            frame.to_csv(output / name, index=False)
        completed = len(all_paths)
        failed = sum(state.status == "failed" for state in statuses)
        state = (("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed") if failure else
                 ("complete" if completed == len(paths) else ("failed" if failed else "pending")))
        result = CollectionManifest(
            schema=1, preparation_fingerprint=configuration.preparation_fingerprint,
            endpoint_complement=endpoint_complement, preparation_sha256=sha(preparation / "preparation.json"),
            status=state, planned_paths=len(paths), selected_paths=len(selected_paths), completed_paths=completed,
            pending_paths=len(paths) - completed - failed, failed_paths=failed, points=len(all_points),
            paths=tuple(statuses), cache=cache_counts,
            sqlite_tables={name: fingerprint(encode_document(writer.get("tables", name)))
                           for name in ("indices.csv.gz", "pathways.csv")},
            files={name: sha(output / name) for name in ("indices.csv.gz", "pathways.csv", "configuration.json")},
        )
        atomic_json(output / "collection.json", result)
        progress(state)
        if failure:
            raise failure
        return result
