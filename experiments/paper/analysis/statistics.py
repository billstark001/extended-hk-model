"""Recoverable typed analysis with immutable protocol and raw-source identities."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_plan
from common.io.json import canonical, fingerprint, sha
from ehk.domain import InputIdentity, FileIdentity
from ehk.metrics.protocol import METRIC_REVISION, metric_protocol
from .models import (AnalysisIdentity, AnalysisJob, AnalysisOptions, AnalysisOutcome, AnalysisPlan,
                     AnalysisProtocol, AnalysisSelection, AnalysisTask, ProbeIdentity)
from .records import ScenarioStatistics, StatisticsStore


def source_identity(directory) -> InputIdentity:
    """Snapshot identities without rereading large trajectories on each resume."""
    directory = Path(directory)
    files = []
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.endswith((".log", "-shm")) or path.name.startswith("."):
                continue
            if path.name in ("session.lock", "active.lock", "job.lock", "logfile.log"):
                continue
            info = path.stat()
            if path.name.endswith("-wal") and info.st_size == 0:
                continue
            content_hash = sha(path) if path.name in ("metadata.json", "trajectory.json", "finished.txt") or path.name.startswith("finished-") else None
            files.append(FileIdentity(path=path.relative_to(directory).as_posix(), size=info.st_size, mtime_ns=info.st_mtime_ns, sha256=content_hash))
    return InputIdentity(directory=str(directory.resolve()), files=tuple(files), fingerprint=fingerprint(files),
                          identity_method="SHA256 metadata/trajectory manifest/completion; size+mtime_ns for bulk raw files")


def analysis_protocol(options: AnalysisOptions | None = None) -> AnalysisProtocol:
    import smp_bindings

    options = options or AnalysisOptions()
    root = Path(__file__).resolve().parents[3]
    sources = [root / name for name in (
        "src/ehk/metrics/protocol.py", "src/ehk/metrics/polarization.py", "src/ehk/metrics/homophily.py",
        "src/ehk/metrics/state.py", "src/ehk/metrics/density.py", "src/ehk/metrics/evaluation.py",
        "src/ehk/metrics/trajectory.py", "src/ehk/micro/observables.py", "src/ehk/domain.py",
        "experiments/paper/analysis/tasks.py", "experiments/paper/analysis/models.py",
        "src/common/numerics/adaptive.py", "src/common/numerics/arrays.py", "src/common/numerics/kde.py",
        "src/common/pipeline.py", "experiments/paper/analysis/statistics.py", "experiments/paper/analysis/records.py",
        "src/common/storage/documents.py",
    )]
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "networkx", "igraph", "leidenalg"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    binding_root = Path(smp_bindings.__file__).resolve().parent
    native_probe = None
    if options.first_passage is not None:
        from experiments.paths import SMP_PROBE_BINARY
        binary = SMP_PROBE_BINARY.resolve()
        native_probe = ProbeIdentity(str(binary), sha(binary))
    return AnalysisProtocol(
        metrics=metric_protocol(endpoint_complement=options.endpoint_complement),
        endpoint_complement=options.endpoint_complement, common_statistics=options.common_statistics,
        metric_revision=options.metric_revision, metadata_source=options.metadata_source,
        first_passage=options.first_passage, opinion_peak_distance=options.opinion_peak_distance,
        active_threshold=options.active_threshold, min_inactive_value=options.min_inactive_value,
        extra_statistics=options.extra_statistics, skip_existing=options.skip_existing,
        analysis_sources={path.relative_to(root).as_posix(): sha(path) for path in sources},
        binding_sources={path.name: sha(path) for path in sorted(binding_root.glob("*.py"))},
        versions=versions, native_probe=native_probe,
    )


@dataclass(frozen=True)
class StatisticsCompute:
    getter: Callable
    scenario_path: str
    origin: str

    def __call__(self, job: AnalysisJob):
        return self.getter(job.metadata, self.scenario_path, self.origin, job.previous)


class StatisticsAdapter(TaskAdapter):
    def __init__(self, store, getter, scenario_path, origin, analysis_id, mode, protocol, ignore_existing, is_complete=None):
        self.store, self.analysis_id, self.mode, self.protocol = store, analysis_id, mode, protocol
        super().__init__(
            lambda task: task.id,
            lambda task: (task.previous is not None and task.previous.complete and not task.previous.stale
                          and (ignore_existing or (is_complete is not None and is_complete(task.previous)))),
            StatisticsCompute(getter, scenario_path, origin), self.persist,
        )

    def persist(self, job: AnalysisJob, result: ScenarioStatistics | None):
        if result is None:
            return False
        source = source_identity(job.source.directory)
        if source != job.source:
            raise RuntimeError(f"raw inputs changed during analysis: {job.id}")
        result.metric_revision = METRIC_REVISION
        result.task_fingerprint = job.task_fingerprint
        result.complete, result.stale = True, False
        self.store.put(result, analysis_id=self.analysis_id, mode=self.mode,
                       metadata=result.metadata or job.metadata, protocol=self.protocol, source=source)
        return True


def analyze_statistics(getter, scenario_base_path, stats_db_path, origin, scenarios, *, mode,
                       protocol: AnalysisOptions | None = None, ignore_exist=True, is_complete=None,
                       concurrency=1, force=False) -> AnalysisOutcome:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("positive analysis concurrency required")
    protocol = analysis_protocol(protocol)
    analysis_id = fingerprint(AnalysisIdentity(mode, protocol))
    tasks = tuple(AnalysisTask(metadata.UniqueName, metadata) for metadata in scenarios)
    database = Path(stats_db_path).resolve()
    control = database.parent / "runtime" / fingerprint(
        AnalysisSelection(str(database), origin, mode, analysis_id, fingerprint(tasks)))[:20]
    with session_lock(control), interruptible():
        workflow = Workflow({}, control / "attempt.json")
        plan = AnalysisPlan(mode, str(Path(scenario_base_path).resolve()), analysis_id, protocol, tasks)
        # Both generic framework adapters below are JSON file write boundaries.
        workflow.stages["plan"] = lambda: freeze_plan(control / "plan.json", json.loads(canonical(plan)))
        workflow.dispatch("plan")
        journal = TaskJournal(control)
        state = "complete"
        try:
            with StatisticsStore(database) as store:
                jobs = []
                for task in tasks:
                    source = source_identity(Path(scenario_base_path) / task.id)
                    task_fingerprint = fingerprint((task.metadata, source))
                    previous = store.get(task.id, origin, analysis_id)
                    if previous is not None and previous.task_fingerprint != task_fingerprint:
                        store.invalidate(task.id, origin, analysis_id,
                                         reason="raw input or requested metadata identity changed", current_source=source)
                        previous = None
                    elif previous is not None and force:
                        store.invalidate(task.id, origin, analysis_id,
                                         reason="forced recomputation pending", current_source=source)
                    jobs.append(AnalysisJob(task.id, task.metadata, previous, source, task_fingerprint))
                adapter = StatisticsAdapter(store, getter, scenario_base_path, origin, analysis_id,
                                            mode, protocol, ignore_exist, is_complete)
                def execute():
                    execute_tasks(jobs, adapter, workers=concurrency, executor="process",
                                  journal=journal, raise_failures=False, force=force)
                    return AnalysisOutcome.from_journal(
                        journal.finish(task.id for task in tasks), analysis_id, database)
                workflow.stages["analyze"] = execute
                return workflow.dispatch("analyze")
        except BaseException as error:
            state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            journal.finish((task.id for task in tasks), state)
