"""Validate binned KDE with exact references recomputed on frozen raw histories."""

import hashlib
import inspect
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import atomic_json, fingerprint, sha
from common.runtime.process import run_tasks
from experiments.paper.metric_comparison import core, raw
from experiments.paper.metric_comparison import report as baseline_reporting
from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.state import MeasurementRecord
from ehk.metrics.density import DensityDiagnostics
from ehk.domain import parse_dataclass, rows_frame
from common.experiments.progress import Progress
from experiments.paper.metric_comparison.contracts import (CollectionConfiguration, CollectionManifest as MainCollection,
    CollectionReceipt, PipelineStatus, ReportManifest, PreparedPath, PathKey, table_records, CollectionProgress, CacheCounts, StatusCounts)
from experiments.paper.recommendation_shared.storage import write_table
from .kernel import KernelDiagnostics
from .contracts import (
    ActivityConfiguration, AdaptiveConfiguration, BaselineEvidence, BaselinePoint, BenchmarkKey,
    CollectionManifest, Configuration, DiagnosticObservation, GroupResult, Job,
    NumericalSources, PathEntry, PathResult, PathRow, PathStatus, PointRow, Receipt, ReportIdentity,
    Selection, TimingRow,
)
from common.storage.documents import DocumentWrite, DocumentWriter, encode_document
from experiments.paper.metric_comparison.receipts import receipt

PATH_KEYS = ("profile", "task_id", "stage", "arm")
DEFAULT_BINS = (1021, 2041, 4081, 8161)
METRICS = ("Ip", "Is", "Ih")


def _calculator():
    from .kernel import BinnedDistanceCalculator

    return BinnedDistanceCalculator()


def _source_configuration():
    import networkx
    import smp_bindings.events_db
    import smp_bindings.model_state
    import smp_bindings.trajectory

    from ehk.micro import observables

    from . import kernel

    sources = {Path(__file__), Path(__file__).with_name("__main__.py"), Path(kernel.__file__),
               Path(inspect.getfile(observables.get_indices)), Path(inspect.getfile(MeasurementEvaluator)),
               Path(smp_bindings.events_db.__file__), Path(smp_bindings.model_state.__file__),
               Path(smp_bindings.trajectory.__file__)}
    return NumericalSources({str(path.resolve()): sha(path) for path in sorted(sources)},
        raw._source_configuration(), "exact",
        {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "networkx": networkx.__version__},
        "linear_binned_gaussian", "absolute completed step", AdaptiveConfiguration(.01, 512, True),
        ActivityConfiguration(.98, .75), "internal explicit exact reference and exact adaptive grid on every selected path",
        "uncached paired same-state exact/binned totals; each includes its own prepare")


def _load_baseline(preparation, baseline):
    points, paths, provenance = baseline_reporting.load_aligned(preparation, baseline / "alignment")
    policies = [core.endpoint_mode(provenance.comparison)]
    manifests = {}
    for directory, name, contract in ((baseline, "run_status.json", PipelineStatus),
            (baseline / "collection", "collection.json", MainCollection),
            (baseline / "report", "report.json", ReportManifest)):
        manifest = core.load_contract(directory / name, contract)
        state = manifest.state if isinstance(manifest, PipelineStatus) else manifest.status
        if state != "complete":
            raise ValueError(f"formal baseline is incomplete: {directory / name}")
        if not isinstance(manifest, PipelineStatus):
            core._authenticate_files(directory, manifest)
        policies.append(core.endpoint_mode(manifest))
        manifests[name] = sha(directory / name)
    source = core.load_contract(baseline / "collection/configuration.json", CollectionConfiguration)
    endpoint_complement = core.endpoint_mode(source)
    if any(policy != endpoint_complement for policy in policies):
        raise ValueError("formal baseline endpoint_complement policies disagree")
    if source.preparation_fingerprint != fingerprint(provenance.preparation):
        raise ValueError("baseline collection belongs to a different preparation")
    if source.metrics != raw._source_configuration():
        raise ValueError("current metric/reader sources differ from the frozen baseline")
    return points, paths, BaselineEvidence(endpoint_complement, sha(preparation / "preparation.json"),
        manifests["run_status.json"], manifests["collection.json"], fingerprint(source),
        sha(baseline / "alignment/comparison.json"), ReportIdentity(str(baseline / "report"), manifests["report.json"]))


def _features(block):
    features = set()
    for row in table_records(block, PreparedPath):
        profile = row.profile
        for name in ("profile", "stage", "arm", "epsilon", "history", "repost", "sensitivity"):
            features.add((name, str((profile, getattr(row, name)))))
        features.add(("setting", str((profile, row.influence, row.rewiring))))
        features.add(("condition", str((profile, row.sensitivity, row.epsilon, row.history))))
        features.add(("terminal", str((profile, row.terminal_status, row.major_components))))
    return features


def select_tasks(paths, points, max_tasks=32, *, full=False):
    """Freeze deterministic coverage of existing groups, never partial task arms."""
    groups = {tuple(key): block for key, block in paths.groupby(["profile", "task_id"], sort=True)}
    feature_sets = {key: _features(block) for key, block in groups.items()}
    # Prioritize groups near the existing activity threshold and a midpoint-error
    # proxy on the saved grid. The proxy is not a reconstructed adaptive decision.
    margins = {}
    for key, block in points.groupby(["profile", "task_id"], sort=True):
        activity, adaptive = float("inf"), float("inf")
        for _, curve in block.groupby(["stage", "arm"], sort=False):
            curve = curve.sort_values("step")
            values = curve[["I_p", "I_s", "I_h"]].to_numpy(float)
            threshold = max(.98 * values[-1, 1], .75)
            activity = min(activity, float(np.min(np.abs(values[:, 1] - threshold))))
            if len(values) > 2:
                steps = curve["step"].to_numpy(float)
                fraction = ((steps[1:-1] - steps[:-2]) / (steps[2:] - steps[:-2]))[:, None]
                residual = np.max(np.abs(values[1:-1] - values[:-2] * (1 - fraction) - values[2:] * fraction), axis=1)
                adaptive = min(adaptive, float(np.min(np.abs(residual - .01))))
        margins[tuple(key)] = activity, adaptive
    for index, field in enumerate(("near_activity", "saved_grid_midpoint_proxy")):
        ordered = sorted(groups, key=lambda key: (margins.get(key, (np.inf, np.inf))[index], key))
        for key in ordered[:min(4, len(ordered))]:
            feature_sets[key].add((field, "selected_stress_group"))
    selected, covered = [], set()
    remaining = set(groups)
    limit = len(groups) if full else min(max_tasks, len(groups))
    while remaining and len(selected) < limit:
        key = min(remaining, key=lambda item: (-len(feature_sets[item] - covered), item))
        selected.append(key)
        covered.update(feature_sets[key])
        remaining.remove(key)
    selected.sort()
    selected_keys = set(selected)
    rows = paths[[key in selected_keys for key in zip(paths["profile"], paths["task_id"])]].copy()
    selection = Selection("full" if full else "representative", None if full else max_tasks,
        len(groups), len(selected), len(paths), len(rows), tuple(selected),
        {profile: sum(key[0] == profile for key in selected) for profile in sorted(paths["profile"].unique())},
        table_records(rows, PathKey), tuple(sorted(covered)),
        tuple(sorted(set().union(*feature_sets.values()) - covered)),
        "greedy existing-design/state coverage; deterministic lexical ties",
        "existing-grid midpoint residual near .01; proxy, not original rejected candidates")
    return rows, selection


def _benchmark_keys(points, paths, count):
    """Select fixed saved observations across task groups before timing begins."""
    metadata = paths[[*PATH_KEYS, "end_step"]]
    candidates = points.merge(metadata, on=list(PATH_KEYS), validate="many_to_one")
    endings = candidates[candidates["step"] == candidates["end_step"]]
    terminal_s = endings[[*PATH_KEYS, "I_s"]].rename(columns={"I_s": "terminal_Is"})
    candidates = candidates.merge(terminal_s, on=list(PATH_KEYS), validate="many_to_one")
    thresholds = np.maximum(.98 * candidates["terminal_Is"].to_numpy(float), .75)
    candidates["activity_margin"] = np.abs(candidates["I_s"].to_numpy(float) - thresholds)

    def ordered():
        # A near-threshold state and an endpoint per initial state precede fillers.
        for _, block in candidates.groupby(["profile", "task_id"], sort=True):
            yield block.sort_values(["activity_margin", *PATH_KEYS, "step"]).iloc[0]
        for _, block in endings.groupby(["profile", "task_id"], sort=True):
            yield block.sort_values([*PATH_KEYS, "step"]).iloc[-1]
        yield from (row for _, row in candidates.sort_values([*PATH_KEYS, "step"]).iterrows())

    result, seen = [], set()
    for row in ordered():
        identity = tuple(row[name] for name in (*PATH_KEYS, "step"))
        if identity not in seen:
            seen.add(identity)
            result.append(BenchmarkKey(*(row[name] for name in PATH_KEYS), int(row["step"])))
        if len(result) >= count:
            break
    return tuple(result)


def _state_key(record, step):
    state, step, epsilon = MeasurementEvaluator._extract(record, step)
    return (step, *MeasurementEvaluator._physical_keys(state, epsilon))


class CandidateEvaluator(MeasurementEvaluator):
    """Canonical scalar caches with per-resolution numerical diagnostics."""

    def __init__(self, bins, shared):
        super().__init__(method="binned", bins=bins, calculator=shared.calculator.reference)
        self.shared = shared
        self.diagnostics = {}
        self._details = OrderedDict()

    def point(self, record, step):
        self._observation = (record.directory, int(step))
        return super().point(record, step)

    def distance(self, record, step):
        return self.point(record, step)[:2]

    def _distance(self, state, step, epsilon, keys):
        prepared, _, input_sha = self.shared.prepare_state(state, step, epsilon)
        key = keys[:2]
        if key not in self._details:
            self._details[key] = (DensityDiagnostics(), DensityDiagnostics())
        details = self._details[key]
        self._details.move_to_end(key)
        def measure(index):
            started = time.perf_counter()
            if index == 0:
                value = self.calculator.opinion_index(prepared.objective, method="binned", bins=self.bins,
                                                      diagnostics=details[0])
            else:
                value = self.calculator.edge_index(prepared.edge, method="binned", bins=self.bins,
                                                   diagnostics=details[1])
            self.shared.evaluate_seconds[self.bins] += time.perf_counter() - started
            return value
        values = (self._opinion.resolve(keys[0], lambda: measure(0)),
                  self._edge.resolve(keys[1], lambda: measure(1)))
        diagnostic = self.shared.calculator._diagnostics(prepared, method="linear_binned_gaussian", bins=self.bins,
            opinion_detail=details[0], edge_detail=details[1])
        self.diagnostics[self._observation] = DiagnosticObservation(diagnostic, input_sha)
        if len(self._details) > 512:
            self._details.popitem(last=False)
        return values


class SharedInputs:
    def __init__(self, calculator, bins):
        self.calculator = calculator
        self.inputs = OrderedDict()
        self.prepare_seconds = 0.
        self.evaluate_seconds = dict.fromkeys(bins, 0.)
        self.benchmarks = OrderedDict()

    def prepare_state(self, state, step, epsilon):
        from ehk.metrics.homophily import mean_state_edge_concordance, normalize_homophily

        key = (step, *MeasurementEvaluator._physical_keys(state, epsilon))
        if key in self.inputs:
            self.inputs.move_to_end(key)
            return self.inputs[key]
        started = time.perf_counter()
        prepared = self.calculator.prepare_state(state, epsilon, step)
        self.prepare_seconds += time.perf_counter() - started
        raw = mean_state_edge_concordance(state, epsilon)
        value = prepared, float(normalize_homophily(raw, epsilon)), _input_sha(key)
        self.inputs[key] = value
        if len(self.inputs) > 16:
            self.inputs.popitem(last=False)
        return value


def _input_sha(key):
    digest = hashlib.sha256()
    for item in key:
        data = item if isinstance(item, bytes) else repr(item).encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _activity(steps, isub):
    from ehk.micro.observables import calc_active_step

    _, step, threshold, _, _ = calc_active_step(np.asarray(steps), np.asarray(isub), .98, .75)
    return int(step), float(threshold)


def _iw(p, h, *, endpoint_complement=True):
    from ehk.metrics.trajectory import pathway_index

    return float(pathway_index(np.asarray(p), np.asarray(h), endpoint_complement=endpoint_complement))


@dataclass(frozen=True, slots=True)
class TimedEvaluation:
    values: tuple[float, float, KernelDiagnostics]
    prepare: float
    evaluate: float
    total: float


def _timing(record, step, bins, repeats, calculator, identity: PathKey):
    graph, opinions, epsilon = record.graph_at(step), record.opinions[step], record.epsilon
    digest = _input_sha(_state_key(record, step))
    rows = []
    for repeat in range(repeats):
        for resolution in bins:
            measured: dict[str, TimedEvaluation] = {}
            for method in (("exact", "binned") if repeat % 2 == 0 else ("binned", "exact")):
                started = time.perf_counter()
                prepared = calculator.prepare(graph, opinions, epsilon, step)
                prepared_at = time.perf_counter()
                values = calculator.exact(prepared) if method == "exact" else calculator.evaluate(prepared, resolution)
                finished_at = time.perf_counter()
                measured[method] = TimedEvaluation(values, prepared_at - started, finished_at - prepared_at, finished_at - started)
            exact, candidate = measured["exact"], measured["binned"]
            rows.append(TimingRow(identity.profile, identity.task_id, identity.stage, identity.arm,
                fingerprint((identity, step)), step, resolution, repeat, digest,
                "exact_first" if repeat % 2 == 0 else "binned_first", exact.prepare, exact.evaluate, exact.total,
                candidate.prepare, candidate.evaluate, candidate.total,
                float(exact.values[0]), float(exact.values[1]), float(candidate.values[0]), float(candidate.values[1]), candidate.values[2]))
    return rows


def _measure_path(entry: PathEntry, bins, repeats, shared, evaluators, *, endpoint_complement=True) -> PathResult:
    from smp_bindings import RawSimulationRecord
    from ehk.metrics.trajectory import sample_indices

    path, baseline_points, baseline_receipt = entry.path, entry.points, entry.baseline
    if baseline_receipt.result.path.endpoint_complement != endpoint_complement:
        raise ValueError("baseline receipt endpoint_complement differs from selected policy")
    identity, directory = path.key, Path(path.raw_directory)
    metadata = raw._validate_raw(directory, path, baseline_points)
    prepared_steps = tuple(point.step for point in baseline_points)
    source_points = {point.step: (point.I_p, point.I_s, point.I_h) for point in baseline_points}
    rows, path_rows, timings = [], [], []
    with raw.PathHeartbeat(identity.label) as heartbeat, RawSimulationRecord(str(directory.parent), metadata) as saved:
        record = MeasurementRecord.from_saved_record(saved)
        for evaluator in evaluators.values():
            evaluator.diagnostics.clear()
        started_reference = time.perf_counter()
        reference = MeasurementEvaluator(method="exact", calculator=shared.calculator.reference)
        heartbeat.phase = "exact reference adaptive grid"
        def exact_point(step):
            heartbeat.sample(step)
            return reference.point(record, int(step))
        baseline_steps, h, p, s = sample_indices(record.max_step, exact_point)
        baseline_steps = tuple(int(step) for step in baseline_steps)
        exact = {step: tuple(float(v) for v in exact_point(step))
                 for step in sorted(set(prepared_steps) | set(baseline_steps))}
        exact_curve = np.array([exact[step] for step in baseline_steps])
        baseline_iw = _iw(p, h, endpoint_complement=endpoint_complement)
        extra_calls = len(set(baseline_steps) - set(prepared_steps))
        extra_seconds = time.perf_counter() - started_reference
        active_step, active_threshold = _activity(baseline_steps, exact_curve[:, 1])
        for resolution, evaluator in evaluators.items():
            started_path = time.perf_counter()
            before_prepare, before_evaluate = shared.prepare_seconds, shared.evaluate_seconds[resolution]
            point_values = {}
            heartbeat.phase = f"B={resolution} frozen grids"
            for grid, steps in (("prepared", prepared_steps), ("baseline_adaptive", baseline_steps)):
                for step in steps:
                    values = evaluator.point(record, step)
                    heartbeat.sample(step)
                    point_values[step] = values
                    reference_values = exact[step]
                    if values[2] != reference_values[2]:
                        raise ValueError("KDE experiment changed exact homophily")
                    detail = evaluator.diagnostics[(record.directory, step)]
                    rows.append(PointRow(identity.profile, identity.task_id, identity.stage, identity.arm,
                        resolution, step, grid, *reference_values, *source_points.get(step, (None, None, None)),
                        *values, detail.diagnostic, detail.same_input_sha))
            heartbeat.phase = f"B={resolution} independent adaptive grid"
            def candidate_point(step):
                heartbeat.sample(step)
                return evaluator.point(record, int(step))
            steps, h, p, s = sample_indices(record.max_step, candidate_point)
            for index, step in enumerate(steps):
                step = int(step)
                reference_values = exact.get(step, (None, None, float(h[index])))
                detail = evaluator.diagnostics[(record.directory, step)]
                rows.append(PointRow(identity.profile, identity.task_id, identity.stage, identity.arm,
                    resolution, step, "candidate_adaptive", *reference_values, None, None, None,
                    float(p[index]), float(s[index]), float(h[index]), detail.diagnostic, detail.same_input_sha))
            fixed = np.array([point_values[step] for step in baseline_steps])
            fixed_active, fixed_threshold = _activity(baseline_steps, fixed[:, 1])
            adaptive_active, adaptive_threshold = _activity(steps, s)
            path_rows.append(PathRow(identity.profile, identity.task_id, identity.stage, identity.arm,
                resolution, path.start_step, path.end_step, endpoint_complement, baseline_iw,
                baseline_receipt.result.path.I_w, _iw(fixed[:, 0], fixed[:, 2], endpoint_complement=endpoint_complement),
                _iw(p, h, endpoint_complement=endpoint_complement), json.dumps(baseline_steps),
                json.dumps([int(step) for step in steps]), active_step, fixed_active, adaptive_active,
                active_threshold, fixed_threshold, adaptive_threshold, "measured", extra_calls, extra_seconds,
                shared.prepare_seconds - before_prepare, shared.evaluate_seconds[resolution] - before_evaluate,
                time.perf_counter() - started_path))
        heartbeat.phase = "paired kernel timing"
        for step in entry.benchmark_steps:
            key = _state_key(record, step)
            if key not in shared.benchmarks:
                # Only complete paired measurements enter the worker-local cache.
                shared.benchmarks[key]=tuple(_timing(record,step,bins,repeats,shared.calculator,identity))
                if len(shared.benchmarks)>64:
                    shared.benchmarks.popitem(last=False)
            shared.benchmarks.move_to_end(key)
            # The receipt owns the result even when an earlier path never committed.
            timings.extend(shared.benchmarks[key])
            heartbeat.sample(step)
    return PathResult(tuple(rows), tuple(path_rows), tuple(timings))


@dataclass(slots=True)
class WorkerContext:
    configuration_fingerprint: str
    bins: tuple[int,...]
    shared: SharedInputs
    evaluators: dict[int,CandidateEvaluator]


_WORKER_CONTEXT: WorkerContext | None = None


def _collect_group(job: Job) -> GroupResult:
    global _WORKER_CONTEXT
    if (_WORKER_CONTEXT is None or _WORKER_CONTEXT.configuration_fingerprint != job.configuration_fingerprint
            or _WORKER_CONTEXT.bins != job.bins):
        shared = SharedInputs(_calculator(), job.bins)
        _WORKER_CONTEXT = WorkerContext(job.configuration_fingerprint, job.bins, shared,
            {bins: CandidateEvaluator(bins, shared) for bins in job.bins})
    shared, evaluators = _WORKER_CONTEXT.shared, _WORKER_CONTEXT.evaluators
    statuses, receipts = [], []
    for entry in job.entries:
        identity, directory = entry.path.key, Path(entry.path.raw_directory)
        saved = receipt(job.output, "kde_approximation.paths", identity, Receipt)
        missing = raw._required_files(directory)
        if missing:
            statuses.append(PathStatus(identity.profile, identity.task_id, identity.stage, identity.arm, "pending", missing))
            continue
        source = raw._raw_sources(directory)
        if source != entry.baseline.raw_sha256:
            raise ValueError("raw inputs differ from the frozen formal baseline")
        if saved is not None:
            if (saved.configuration_fingerprint != job.configuration_fingerprint or saved.identity != identity
                    or saved.raw_sha256 != source or saved.baseline_receipt_sha256 != entry.baseline_receipt_sha256):
                raise ValueError("approximation receipt differs from the frozen experiment")
            statuses.append(PathStatus(identity.profile, identity.task_id, identity.stage, identity.arm, "cached"))
            continue
        try:
            result = _measure_path(entry, job.bins, job.repeats, shared, evaluators, endpoint_complement=job.endpoint_complement)
            if raw._raw_sources(directory) != source:
                raise ValueError("raw source files changed during approximation calculation")
            receipts.append(Receipt(job.configuration_fingerprint, identity, source, entry.baseline_receipt_sha256, result))
            statuses.append(PathStatus(identity.profile, identity.task_id, identity.stage, identity.arm, "complete"))
        except Exception as error:
            statuses.append(PathStatus(identity.profile, identity.task_id, identity.stage, identity.arm, "failed", error=str(error)))
    return GroupResult(tuple(statuses), tuple(receipts))


def _measurement_frame(rows, kind):
    """Flatten the declared diagnostics at the scientific table export boundary."""
    frame = rows_frame(rows, kind)
    if "diagnostics" in frame:
        detail = rows_frame([row.diagnostics for row in rows], KernelDiagnostics).drop(columns="bins")
        frame = frame.drop(columns="diagnostics").join(detail)
    return frame


def collect(preparation, baseline, output, *, bins=DEFAULT_BINS, max_tasks=32, full=False,
            benchmark_states=64, repeats=3, workers=1, endpoint_complement=True):
    if (type(workers) is not int or not 1 <= workers <= 8 or type(max_tasks) is not int or max_tasks < 1
            or type(benchmark_states) is not int or benchmark_states < 1 or type(repeats) is not int or repeats < 1):
        raise ValueError("workers must be 1..8; task, benchmark and repeat counts must be positive integers")
    bins = tuple(bins)
    if not bins or len(set(bins)) != len(bins) or any(type(value) is not int or value not in DEFAULT_BINS for value in bins):
        raise ValueError("bins must be distinct supported resolutions")
    preparation, baseline, output = (Path(path).expanduser().resolve() for path in (preparation, baseline, output))
    if any(output == source or output in source.parents or source in output.parents for source in (preparation, baseline)):
        raise ValueError("approximation output must be separate from frozen evidence")
    points, paths, evidence = _load_baseline(preparation, baseline)
    if evidence.endpoint_complement != endpoint_complement:
        raise ValueError("frozen baseline endpoint_complement differs from selected policy")
    from ehk.metrics.protocol import metric_protocol
    selected, selection = select_tasks(paths, points, max_tasks, full=full)
    keys = set(selection.path_keys)
    selected_points = points[[PathKey(*key) in keys for key in points[list(PATH_KEYS)].itertuples(index=False, name=None)]]
    benchmarks = _benchmark_keys(selected_points, selected, benchmark_states)
    configuration = Configuration(schema=1, preparation=str(preparation), baseline=str(baseline),
        endpoint_complement=endpoint_complement, preparation_sha256=evidence.preparation_sha256,
        run_status_sha256=evidence.run_status_sha256, collection_sha256=evidence.collection_sha256,
        collection_configuration_fingerprint=evidence.collection_configuration_fingerprint,
        comparison_sha256=evidence.comparison_sha256, baseline_report=evidence.baseline_report,
        selection=selection, selected_paths=selection.path_keys, bins=bins, benchmark_states=benchmarks,
        repeats=repeats, metrics=_source_configuration(), metric_protocol=metric_protocol(endpoint_complement=endpoint_complement),
        exact_reference="numerical study computes method=exact with the canonical formula and independent adaptive grid")
    configuration_fingerprint = fingerprint(configuration)
    groups = selected_points.groupby(list(PATH_KEYS), sort=False).indices
    outcomes: dict[PathKey,PathStatus] = {}
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    terminal = Progress(len(selected), stream=sys.stderr)

    def jobs():
        for path in table_records(selected, PreparedPath):
            saved = receipt(baseline / "collection", "metric_comparison.paths", path.identity, CollectionReceipt)
            if saved is None:
                raise ValueError("formal baseline SQLite path receipt is missing")
            if saved.identity != path.identity or saved.configuration_fingerprint != evidence.collection_configuration_fingerprint:
                raise ValueError("path receipt differs from the authenticated baseline")
            key = (path.profile, path.task_id, path.stage, path.arm)
            entry = PathEntry(path, table_records(selected_points.iloc[groups[key]], BaselinePoint), saved,
                fingerprint(saved), tuple(row.step for row in benchmarks
                    if PathKey(row.profile,row.task_id,row.stage,row.arm) == path.key))
            yield Job((path.profile,path.task_id), bins, repeats, endpoint_complement, configuration_fingerprint, (entry,), output)

    def progress(state):
        counts = StatusCounts.count(outcomes.values())
        atomic_json(output / "progress.json", CollectionProgress(state,workers,len(selected),len(selected),
            counts.complete + counts.cached,counts,time.monotonic() - started,CacheCounts()))

    def finished(job: Job, result: GroupResult):
        writer.put_many(DocumentWrite("kde_approximation.paths", fingerprint(saved.identity), saved) for saved in result.receipts)
        for row in result.paths:
            key = PathKey(row.profile,row.task_id,row.stage,row.arm)
            outcomes[key] = row
            terminal.event(key.label,"deferred" if row.status == "pending" else row.status)
        progress("running")

    failure = None
    with session_lock(output / "runtime"), interruptible(), DocumentWriter(output / "analysis.sqlite") as writer:
        freeze_manifest(output / "configuration.json", configuration)
        progress("running")
        try:
            if workers == 1:
                for job in jobs():
                    terminal.event(job.entries[0].path.key.label,"running")
                    finished(job,_collect_group(job))
            else:
                run_tasks(_collect_group,jobs(),workers,executor="process",on_result=finished,
                    event=lambda job,state,**details: terminal.event(job.entries[0].path.key.label,state))
        except BaseException as error:
            failure = error
        previous_path = output / "collection.json"
        if failure is None and len(outcomes) == len(selected) and all(item.status == "cached" for item in outcomes.values()) and previous_path.is_file():
            previous = core.load_contract(previous_path,CollectionManifest)
            if previous.status == "complete" and previous.configuration_sha256 == sha(output / "configuration.json"):
                core._authenticate_files(output,previous)
                progress("complete")
                return previous
        point_rows,path_rows,timing_rows,statuses = [],[],[],[]
        for identity in selection.path_keys:
            document = writer.get("kde_approximation.paths",fingerprint(identity))
            saved = None if document is None else parse_dataclass(Receipt,document)
            state = outcomes.get(identity,PathStatus(identity.profile,identity.task_id,identity.stage,identity.arm,"pending"))
            if saved is not None:
                if saved.configuration_fingerprint != configuration_fingerprint or saved.identity != identity:
                    raise ValueError("receipt does not belong to this approximation configuration")
                point_rows.extend(saved.result.points)
                path_rows.extend(saved.result.paths)
                timing_rows.extend(saved.result.benchmarks)
                if state.status == "pending":
                    state = replace(state,status="complete")
            statuses.append(state)
        for name,frame in (("points.csv.gz",_measurement_frame(point_rows,PointRow)),
                ("paths.csv.gz",rows_frame(path_rows,PathRow)),("benchmarks.csv",_measurement_frame(timing_rows,TimingRow))):
            if name == "benchmarks.csv" and len(frame):
                frame = frame.drop_duplicates(["same_input_sha","repeat","bins"])
            write_table(output,name,frame,writer=writer)
            frame.to_csv(output / name,index=False)
        complete = sum(row.status in ("complete","cached") for row in statuses)
        failed = sum(row.status == "failed" for row in statuses)
        state = (("interrupted" if isinstance(failure,KeyboardInterrupt) else "failed") if failure else
                 ("complete" if complete == len(selected) else ("failed" if failed else "pending")))
        manifest = CollectionManifest(schema=1,status=state,planned_paths=len(selected),completed_paths=complete,
            endpoint_complement=endpoint_complement,pending_paths=len(selected)-complete-failed,failed_paths=failed,
            selection=selection,paths=tuple(statuses),configuration_sha256=sha(output / "configuration.json"),
            sqlite_tables={name:fingerprint(encode_document(writer.get("tables",name)))
                for name in ("points.csv.gz","paths.csv.gz","benchmarks.csv")},
            files={name:sha(output / name) for name in ("points.csv.gz","paths.csv.gz","benchmarks.csv")})
        atomic_json(output / "collection.json",manifest)
        progress(state)
        if failure:
            raise failure
        return manifest
