"""Paired process controls with resumable SQLite analysis and one metric core."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.experiments.progress import Progress
from common.io.json import fingerprint, sha
from common.storage.documents import DocumentStore, DocumentWriter
from ehk.domain import FullResult, freeze_plan, parse_dataclass, parse_metadata, read_plan, rows_frame
from ehk.micro.persistent import observe_persistent
from experiments.paper.analysis.models import AnalysisOptions
from experiments.paper.analysis.statistics import analysis_protocol, source_identity
from experiments.paper.recommendation_shared.storage import write_table
from ..recommendation_shared import runtime
from ..study import execution_provenance
from .contracts import (AnalysisSummary, ArmIdentity, ArmJob, ArmMeasurement, IndexRow, IntegralRow,
                        MetricReceipt, SnapshotEntry, SnapshotIndex, SnapshotInput, SnapshotPosts)
from .design import ARMS, ControlDesign, arm_metadata


def freeze(directory, binary, design: ControlDesign | None = None):
    directory = Path(directory).resolve()
    plan = freeze_plan(directory / "plan.json", (design or ControlDesign()).plan())
    freeze_manifest(directory / "runtime" / "binding.json", execution_provenance(binary))
    return plan


def _checkpoint(directory, task, arm, step):
    return directory / "raw" / arm_metadata(task, arm).UniqueName / f"checkpoint-{step:09d}.msgpack.lz4"


def _physical(path):
    from smp_bindings.model_state import load_snapshot
    return load_snapshot(str(path))


def _initial_pair(directory, task, arms):
    initial = None
    for arm in arms:
        state = _physical(_checkpoint(directory, task, arm, 0))
        if initial is None:
            initial = state
        elif state != initial:
            raise ValueError(f"process controls have different initial microscopic/RNG states: {task.id}")


def run_study(directory, binary, workers=4, design: ControlDesign | None = None):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("controls support 1 through 6 concurrent pairs")
    plan = read_plan(directory / "plan.json") if (directory / "plan.json").is_file() and design is None else freeze(directory, binary, design)
    if plan.study != "paper_pathway_controls":
        raise ValueError("directory contains a different experiment")

    def run_task(task, frozen, binding, previous, progress):
        arms = dict(previous.arms) if previous is not None else {}
        if arms.keys() - set(ARMS):
            raise ValueError("record contains an undeclared process control")
        for arm in ARMS:
            if arm not in arms or not runtime.complete_raw(directory, arms[arm]):
                result = observe_persistent(binary, directory / "raw", arm_metadata(task, arm),
                                            frozen.protocol.snapshot_steps, fixed_horizon=True)
                raw = Path(result.run_directory)
                result.run_directory = None
                result.raw_record = raw.relative_to(directory).as_posix()
                result.binary_sha256 = binding.binary_sha256
                arms[arm] = result
            _initial_pair(directory, task, arms)
            progress(FullResult(arms=arms))
        return FullResult(arms=arms)

    return runtime.run_dataset(directory, binary, workers, plan_factory=lambda: plan,
        binding_factory=lambda plan, path, binary_sha: execution_provenance(binary),
        task_runner=run_task, result_arms=lambda result: result.arms.values(), resume=True,
        cache_valid=lambda result, binding: (set(result.arms) == set(ARMS) and
            all(arm.binary_sha256 == binding.binary_sha256 for arm in result.arms.values())))


def snapshot_input(path) -> SnapshotInput:
    """Extract complete directed edges, node opinions and retained post records."""
    state = _physical(path)
    data = state.data
    edges = tuple(sorted((int(source), int(target)) for source, targets in data.Graph.adjacency.items() for target in targets))
    return SnapshotInput(1, data.CurStep - 1, state.dynamics_type, tuple(range(len(data.Opinions))),
                         data.Opinions, edges,
                         tuple(SnapshotPosts(int(sharer), entries) for sharer, entries in sorted(data.Posts.items())))


def export_snapshots(directory) -> SnapshotIndex:
    directory = Path(directory).resolve()
    plan = read_plan(directory / "plan.json")
    rows = []
    meter = Progress(len(plan.tasks))
    with DocumentStore(directory / "analysis.sqlite") as store:
        previous = store.get("controls.index", plan.plan_fingerprint)
        retained = {} if previous is None else {row.snapshot_key: row for row in parse_dataclass(SnapshotIndex, previous).snapshots}
        for task in plan.tasks:
            meter.event(task.id, "running")
            _initial_pair(directory, task, ARMS)
            for arm in ARMS:
                for step in plan.protocol.snapshot_steps:
                    source = _checkpoint(directory, task, arm, step)
                    source_hash = sha(source)
                    key = fingerprint([plan.plan_fingerprint, task.id, arm, step])
                    cached = retained.get(key)
                    if cached is not None and cached.checkpoint_sha256 == source_hash:
                        saved = store.get("controls.snapshots", key)
                        if saved is not None and fingerprint(parse_dataclass(SnapshotInput, saved)) == cached.state_fingerprint:
                            rows.append(cached)
                            continue
                    snapshot = snapshot_input(source)
                    if sha(source) != source_hash:
                        raise RuntimeError(f"control checkpoint changed during export: {source}")
                    if snapshot.step != step:
                        raise ValueError(f"checkpoint has the wrong completed step: {source}")
                    store.put("controls.snapshots", key, snapshot)
                    rows.append(SnapshotEntry(task.id, task.coordinates.case, task.coordinates.replicate, arm, step,
                        source.relative_to(directory).as_posix(), source_hash, key, fingerprint(snapshot)))
            meter.event(task.id, "complete")
        index = SnapshotIndex(1, plan.plan_fingerprint, tuple(rows))
        store.put("controls.index", plan.plan_fingerprint, index)
    return index


def _compute_arm(job: ArmJob) -> ArmMeasurement:
    import numpy as np
    from smp_bindings import RawSimulationRecord
    from ehk.micro.observables import get_indices
    from ehk.metrics.trajectory import pathway_index

    raw = Path(job.dataset) / "raw"
    path = raw / job.metadata.UniqueName / "metadata.json"
    actual = parse_metadata(json.loads(path.read_text()))
    for name in ("RNG", "HKParams", "NodeCount", "RecsysFactoryType", "RecSysParams"):
        if getattr(actual, name) != getattr(job.metadata, name):
            raise ValueError(f"raw control metadata differs in {name}: {path}")
    with RawSimulationRecord(str(raw), actual) as record:
        if not record.is_finished or not record.is_sanitized:
            raise ValueError(f"control record is incomplete or missing required channels: {path}")
        if record.max_step != job.horizon:
            raise ValueError(f"control record did not reach its frozen horizon: {path}")
        steps, h, p, s = get_indices(record)
        rows = tuple(IndexRow(job.identity.pair_id, job.case, job.replicate, job.identity.arm,
                              int(step), float(ip), float(is_), float(ih) if np.isfinite(ih) else None)
                     for step, ih, ip, is_ in zip(steps, h, p, s, strict=True))
        integral = IntegralRow(job.identity.pair_id, job.case, job.replicate, job.identity.arm,
            record.max_step, float(pathway_index(p, h, endpoint_complement=job.endpoint_complement)), job.endpoint_complement)
    if source_identity(job.identity.source.directory) != job.identity.source:
        raise RuntimeError(f"control raw inputs changed during analysis: {job.id}")
    return ArmMeasurement(job.identity, rows, integral)


def analyze(directory, *, endpoint_complement=True, concurrency=1, force=False):
    """Resume verified per-arm results; parallel workers never open the output DB."""
    from ehk.micro import observables
    from ehk.metrics.protocol import metric_protocol

    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("positive analysis concurrency required")
    directory = Path(directory).resolve()
    output = directory / "analysis"
    with session_lock(output / "runtime"), interruptible():
        index = export_snapshots(directory)
        plan = read_plan(directory / "plan.json")
        protocol = analysis_protocol(AnalysisOptions(endpoint_complement=endpoint_complement))
        root = Path(__file__).resolve().parents[3]
        own_sources = (Path(__file__), Path(__file__).with_name("contracts.py"), Path(__file__).with_name("design.py"))
        protocol = replace(protocol, analysis_sources={
            **protocol.analysis_sources, **{path.relative_to(root).as_posix(): sha(path) for path in own_sources}
        })
        jobs = []
        for task in plan.tasks:
            for arm in ARMS:
                metadata = arm_metadata(task, arm)
                identity = ArmIdentity(plan.plan_fingerprint, task.id, arm,
                                       source_identity(directory / "raw" / metadata.UniqueName), protocol)
                jobs.append(ArmJob(str(directory), metadata, task.coordinates.case, task.coordinates.replicate,
                                   plan.protocol.horizon, endpoint_complement, identity, fingerprint(identity)))
        completed: dict[str, ArmMeasurement] = {}
        with DocumentWriter(output / "analysis.sqlite") as store:
            def cached(job):
                saved = store.get("controls.arm_metrics", job.key)
                if saved is None:
                    return False
                record = parse_dataclass(ArmMeasurement, saved)
                if record.identity != job.identity:
                    raise ValueError(f"control cached identity differs: {job.id}")
                completed[job.id] = record
                return True
            def save(job, record):
                store.put("controls.arm_metrics", job.key, record)
                completed[job.id] = record
            execute_tasks(jobs, TaskAdapter(lambda job: job.id, cached, _compute_arm, save),
                          workers=concurrency, executor="process" if concurrency > 1 else "thread", force=force)
            rows = [row for job in jobs for row in completed[job.id].indices]
            summaries = [completed[job.id].integral for job in jobs]
            receipt = MetricReceipt(plan.plan_fingerprint, len(index.snapshots), sha(observables.__file__), None,
                ("I_p", "I_s", "I_h", "I_w"), endpoint_complement,
                metric_protocol(endpoint_complement=endpoint_complement), protocol)
            store.put("controls.receipts", fingerprint([plan.plan_fingerprint, protocol]), receipt)
        for name, values in (("indices.csv", rows), ("pathway_integrals.csv", summaries)):
            frame = rows_frame(values)
            write_table(output, name, frame)
            frame.to_csv(output / name, index=False)
        return AnalysisSummary(len(index.snapshots), len(rows), len(summaries))


def plot_snapshots(directory, output=None):
    """Preview persisted network inputs with one layout shared by every arm."""
    import matplotlib.pyplot as plt
    import networkx as nx

    directory = Path(directory).resolve()
    plan = read_plan(directory / "plan.json")
    output = Path(output).resolve() if output is not None else directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    with DocumentStore(directory / "analysis.sqlite", readonly=True) as store:
        wire_index = store.get("controls.index", plan.plan_fingerprint)
        if wire_index is None:
            raise ValueError("missing analyzed snapshot inputs; run pathway_controls analyze first")
        index = parse_dataclass(SnapshotIndex, wire_index)
        for pair in sorted({row.pair_id for row in index.snapshots}):
            rows = [row for row in index.snapshots if row.pair_id == pair]
            first = parse_dataclass(SnapshotInput, store.get("controls.snapshots", rows[0].snapshot_key))
            graph = nx.DiGraph()
            graph.add_nodes_from(first.node_ids)
            graph.add_edges_from(first.edges)
            positions = nx.spring_layout(graph.to_undirected(), seed=0)
            steps = sorted({row.step for row in rows})
            figure, axes = plt.subplots(len(ARMS), len(steps), squeeze=False, figsize=(3 * len(steps), 8))
            for row in rows:
                snapshot = parse_dataclass(SnapshotInput, store.get("controls.snapshots", row.snapshot_key))
                graph = nx.DiGraph()
                graph.add_nodes_from(snapshot.node_ids)
                graph.add_edges_from(snapshot.edges)
                axis = axes[ARMS.index(row.arm), steps.index(row.step)]
                nx.draw_networkx(graph, pos=positions, ax=axis, node_color=snapshot.opinions, cmap="coolwarm",
                    vmin=-1, vmax=1, node_size=12, width=.2, alpha=.7, arrows=False, with_labels=False)
                axis.set_title(f"{row.arm}, step {row.step}")
                axis.set_axis_off()
            figure.suptitle(pair)
            figure.tight_layout()
            figure.savefig(output / (pair + ".png"), dpi=140)
            plt.close(figure)
    return output
