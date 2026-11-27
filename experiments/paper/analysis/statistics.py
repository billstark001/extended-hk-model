"""Recoverable per-task analysis with immutable protocol and raw-source identities."""

import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_plan
from common.io.json import fingerprint, sha
from ehk.metrics.protocol import METRIC_REVISION, metric_protocol

from .records import StatisticsStore


def source_identity(directory):
    """Snapshot input identities without rereading large trajectories on each resume."""
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
                continue  # SQLite may create an empty read-side WAL when opening events.db.
            item = {"path": str(path.relative_to(directory)), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
            if path.name in ("metadata.json", "trajectory.json", "finished.txt") or path.name.startswith("finished-"):
                item["sha256"] = sha(path)
            files.append(item)
    return {
        "directory": str(directory.resolve()),
        "files": files,
        "fingerprint": fingerprint(files),
        "identity_method": "SHA256 metadata/trajectory manifest/completion; size+mtime_ns for bulk raw files",
    }


def analysis_protocol(protocol=None):
    import smp_bindings

    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "src/ehk/metrics/protocol.py",
        root / "src/ehk/metrics/polarization.py",
        root / "src/ehk/metrics/homophily.py",
        root / "src/ehk/metrics/state.py",
        root / "src/ehk/metrics/density.py",
        root / "src/ehk/metrics/evaluation.py",
        root / "src/ehk/metrics/trajectory.py",
        root / "src/ehk/micro/observables.py",
        Path(__file__).with_name("tasks.py"),
        root / "src/common/numerics/adaptive.py",
        root / "src/common/numerics/arrays.py",
        root / "src/common/numerics/kde.py",
        root / "src/common/pipeline.py",
        Path(__file__),
        Path(__file__).with_name("records.py"),
        root / "src/common/storage/documents.py",
    ]
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "networkx", "igraph", "leidenalg"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    binding_root = Path(smp_bindings.__file__).resolve().parent
    options = dict(protocol or {})
    endpoint_complement = options.get("endpoint_complement", True)
    evidence = {
        **options,
        "metrics": metric_protocol(endpoint_complement=endpoint_complement),
        "endpoint_complement": endpoint_complement,
        "analysis_sources": {str(path.relative_to(root)): sha(path) for path in sources},
        "binding_sources": {path.name: sha(path) for path in sorted(binding_root.glob("*.py"))},
        "versions": versions,
    }
    if evidence.get("first_passage"):
        from experiments.paths import SMP_PROBE_BINARY

        binary = SMP_PROBE_BINARY.resolve()
        evidence["native_probe"] = {"path": str(binary), "sha256": sha(binary)}
    return evidence


@dataclass
class StatisticsCompute:
    getter: object
    scenario_path: str
    origin: str

    def __call__(self, job):
        return self.getter(job["metadata"], self.scenario_path, self.origin, job["previous"])


class StatisticsAdapter(TaskAdapter):
    def __init__(
        self, store, getter, scenario_path, origin, analysis_id, mode, protocol, ignore_existing, is_complete=None
    ):
        self.store, self.analysis_id, self.mode, self.protocol = store, analysis_id, mode, protocol
        self.ignore_existing = ignore_existing
        super().__init__(
            lambda task: task["id"],
            lambda task: (
                task["previous"] is not None
                and getattr(task["previous"], "complete", False)
                and not getattr(task["previous"], "stale", False)
                and (ignore_existing or (is_complete is not None and is_complete(task["previous"])))
            ),
            StatisticsCompute(getter, scenario_path, origin),
            self.persist,
        )

    def persist(self, job, result):
        if result is None:
            return False
        source = source_identity(Path(job["source"]["directory"]))
        if source != job["source"]:
            raise RuntimeError(f"raw inputs changed during analysis: {job['id']}")
        result.metric_revision = METRIC_REVISION
        result.task_fingerprint = job["task_fingerprint"]
        result.complete = True
        result.stale = False
        self.store.put(
            result,
            analysis_id=self.analysis_id,
            mode=self.mode,
            metadata=getattr(result, "metadata", job["metadata"]),
            protocol=self.protocol,
            source=source,
        )
        return True


def analyze_statistics(
    getter,
    scenario_base_path,
    stats_db_path,
    origin,
    scenarios,
    *,
    mode,
    protocol=None,
    ignore_exist=True,
    is_complete=None,
    concurrency=1,
    force=False,
):
    protocol = analysis_protocol(protocol)
    analysis_id = fingerprint({"mode": mode, "protocol": protocol})
    tasks = [{"id": metadata["UniqueName"], "metadata": metadata} for metadata in scenarios]
    database = Path(stats_db_path).resolve()
    control = (
        database.parent
        / "runtime"
        / fingerprint(
            {
                "database": str(database),
                "origin": origin,
                "mode": mode,
                "analysis_id": analysis_id,
                "selection": fingerprint(tasks),
            }
        )[:20]
    )
    with session_lock(control), interruptible():
        workflow = Workflow({}, control / "attempt.json")
        workflow.stages["plan"] = lambda: freeze_plan(
            control / "plan.json",
            {
                "schema": 3,
                "study": "paper_statistics",
                "profile": mode,
                "workspace": str(Path(scenario_base_path).resolve()),
                "analysis_id": analysis_id,
                "analysis_protocol": protocol,
                "tasks": tasks,
            },
        )
        workflow.dispatch("plan")
        journal = TaskJournal(control)
        state = "complete"
        try:
            with StatisticsStore(database) as store:
                jobs = []
                for task in tasks:
                    source = source_identity(Path(scenario_base_path) / task["id"])
                    task_fingerprint = fingerprint({"metadata": task["metadata"], "source": source})
                    previous = store.get(task["id"], origin, analysis_id)
                    if previous is not None and getattr(previous, "task_fingerprint", None) != task_fingerprint:
                        store.invalidate(
                            task["id"],
                            origin,
                            analysis_id,
                            reason="raw input or requested metadata identity changed",
                            current_source=source,
                        )
                        previous = None
                    elif previous is not None and force:
                        store.invalidate(
                            task["id"],
                            origin,
                            analysis_id,
                            reason="forced recomputation pending",
                            current_source=source,
                        )
                    jobs.append({**task, "previous": previous, "source": source, "task_fingerprint": task_fingerprint})
                adapter = StatisticsAdapter(
                    store, getter, scenario_base_path, origin, analysis_id, mode, protocol, ignore_exist, is_complete
                )

                def execute():
                    execute_tasks(
                        jobs,
                        adapter,
                        workers=max(1, concurrency),
                        executor="process",
                        journal=journal,
                        raise_failures=False,
                        force=force,
                    )
                    return {**journal.finish(tasks), "analysis_id": analysis_id, "database": str(database)}

                workflow.stages["analyze"] = execute
                return workflow.dispatch("analyze")
        except BaseException as error:
            state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            journal.finish(tasks, state)
