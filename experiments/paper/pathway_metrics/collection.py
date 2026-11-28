"""Compare saved continuous I_w tendencies with separately measured P/E histories.

The primary statistics database is read-only. This command measures no I_w and
does not treat P/E threshold order as a true label for continuous tendencies.
Completed P/E histories and their source identities resume from a separate
SQLite document store. Selection is explicit, including the source batch.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import smp_bindings
from smp_bindings import RawSimulationRecord
from smp_bindings.domain import ProbeResponse, parse_dataclass
from smp_bindings.probe import FrozenSimulationState, run_probe_jobs

from common.io.json import canonical, fingerprint, sha
from common.storage.documents import DocumentStore
from ehk.domain import ScenarioMetadata
from ehk.metrics.pathway import (
    EXCESS_THRESHOLD, OBSERVATION_INTERVAL, SEPARATION_THRESHOLD, first_passage,
)
from ehk.metrics.protocol import METRIC_REVISION, metric_protocol
from experiments.paper.analysis.records import StatisticsStore
from experiments.paper.analysis.statistics import source_identity

NAMESPACE = "ehk-iw-pe-comparison-v1"
PLAN_NAMESPACE = "ehk-iw-pe-comparison-plans-v1"


@dataclass(frozen=True)
class RunIdentity:
    origin: str
    name: str


@dataclass(frozen=True)
class ComparisonProtocol:
    schema: int
    observation_interval: int
    include_terminal: bool
    separation_threshold: float
    excess_threshold: float
    input_reader: str
    inline_batch_size: int
    interpretation: str
    probe_binary: str
    probe_sha256: str
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class PrimaryReference:
    identity: RunIdentity
    analysis_id: str
    database: str
    source_fingerprint: str
    raw_directory: str
    metadata: ScenarioMetadata
    metadata_provenance: str | None
    rng_provenance: str | None
    terminal_step: int
    I_w: float
    endpoint_complement: bool
    metric_revision: int
    terminal_I_p: float
    terminal_I_s: float
    terminal_I_h: float
    opinion_peaks: int | None


@dataclass(frozen=True)
class Observation:
    step: int
    P: float
    E: float
    C_edge: float
    C_pair: float


@dataclass(frozen=True)
class EventReaderEvidence:
    tables: dict[str, str]
    copied_events: int
    copied_rewiring_bodies: int
    body_header_actor_disagreements: int | None
    copy_seconds: float
    index_seconds: float


@dataclass(frozen=True)
class ComparisonResult:
    schema: int
    state: str
    protocol_id: str
    primary: PrimaryReference
    observations: tuple[Observation, ...]
    observed_order: str | None
    separation_step: int | None
    segregation_step: int | None
    elapsed_seconds: float
    error: str | None = None
    verified_graph_snapshots: tuple[int, ...] = ()
    event_reader: EventReaderEvidence | None = None


def make_protocol(binary: Path) -> ComparisonProtocol:
    binding = Path(smp_bindings.__file__).resolve().parent
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__), root / "src/ehk/metrics/pathway.py",
             root / "experiments/paper/analysis/statistics.py",
             root / "src/common/storage/documents.py"]
    files.extend(binding / name for name in
                 ("record.py", "probe.py", "domain.py", "model_state.py", "events_db.py", "trajectory.py", "process.py"))
    return ComparisonProtocol(
        1, OBSERVATION_INTERVAL, True, SEPARATION_THRESHOLD, EXCESS_THRESHOLD,
        "RawSimulationRecord opinions and graph replay use events.agent_id, matching primary I_w; a memory-only backup indexes events(type,step,id); every stored graph is checked against sequential replay; bounded inline_f64 states enter the unchanged native pathway evaluator.",
        16,
        "Continuous I_w and P/E observations describe different aspects of the same saved trajectory; observed threshold order is a separate diagnostic, not tendency ground truth.",
        str(binary), sha(binary), {str(path.resolve()): sha(path) for path in files},
    )


def load_selection(path: Path) -> list[RunIdentity]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != 1 or not isinstance(value.get("tasks"), list):
        raise ValueError("selection requires schema=1 and tasks=[{origin,name}, ...]")
    tasks = [RunIdentity(**{key: task[key] for key in ("origin", "name")}) for task in value["tasks"]]
    if not tasks or any(not task.origin or not task.name for task in tasks):
        raise ValueError("selection identities must be nonempty")
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate (origin,name) in selection")
    return tasks


def load_primary(database: Path, analysis_id: str, tasks: list[RunIdentity]) -> list[PrimaryReference]:
    records = []
    with StatisticsStore(database, readonly=True) as store:
        for identity in tasks:
            row = store.get(identity.name, identity.origin, analysis_id)
            if row is None or not row.complete or row.stale:
                raise ValueError(f"missing or incomplete primary statistics: {identity}")
            if (row.mode != "gradation" or row.metric_revision != METRIC_REVISION
                    or row.protocol is None or row.protocol.metrics != metric_protocol(endpoint_complement=True)
                    or row.endpoint_complement is not True):
                raise ValueError(f"primary statistics do not use the adopted completed-index protocol: {identity}")
            if row.metadata is None or row.source is None or row.step is None:
                raise ValueError(f"primary statistics lack metadata/source/terminal step: {identity}")
            if row.pathway_integral is None or not math.isfinite(row.pathway_integral):
                raise ValueError(f"primary I_w is nonfinite: {identity}")
            records.append(PrimaryReference(
                identity, analysis_id, str(database), row.source.fingerprint, row.source.directory,
                row.metadata, row.metadata_provenance, row.rng_provenance, row.step,
                row.pathway_integral, True, row.metric_revision,
                float(row.p_index[-1]), float(row.g_index[-1]), float(row.h_index[-1]),
                row.last_opinion_peak_count,
            ))
    return records


def result_key(protocol_id: str, reference: PrimaryReference) -> str:
    return fingerprint((protocol_id, reference))


def verify_graph_snapshots(record: RawSimulationRecord) -> tuple[int, ...]:
    """Verify header-authoritative event replay against every saved graph.

    Some historical bodies have a subsequently added DEFAULT 0 actor column.
    The established Python reader uses the event header, as did primary I_w.
    Exact snapshot agreement is required before using that recorded event stream.
    """
    graph = record._load_graph(0)
    previous = 0
    checked = []
    for step in sorted(record.graph_paths):
        if step:
            record._advance_graph(graph, previous, step)
        saved = record._load_graph(step)
        if set(graph.edges()) != set(saved.edges()) or set(graph.nodes()) != set(saved.nodes()):
            raise ValueError(f"event-header graph replay differs from saved graph at step {step}")
        checked.append(step)
        previous = step
    return tuple(checked)


def index_events_in_memory(record: RawSimulationRecord) -> EventReaderEvidence:
    """Accelerate old unindexed logs without writing to their source database."""
    if record.events_db is None:
        raise ValueError("raw event reader is not open")
    memory = sqlite3.connect(":memory:")
    try:
        started = time.monotonic()
        record.events_db.backup(memory)
        copied = time.monotonic()
        memory.execute("CREATE INDEX IF NOT EXISTS ehk_pe_replay_events ON events(type,step,id)")
        indexed = time.monotonic()
        tables = dict(memory.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name IN ('events','rewiring_events')"))
        events = memory.execute("SELECT count(*) FROM events").fetchone()[0]
        bodies = memory.execute("SELECT count(*) FROM rewiring_events").fetchone()[0]
        columns = {row[1] for row in memory.execute("PRAGMA table_info(rewiring_events)")}
        disagreements = memory.execute("SELECT count(*) FROM events e JOIN rewiring_events r ON r.event_id=e.id WHERE e.agent_id!=r.agent_id").fetchone()[0] if "agent_id" in columns else None
        evidence = EventReaderEvidence(tables, events, bodies, disagreements, copied - started, indexed - copied)
    except BaseException:
        memory.close()
        raise
    record.events_db.close()
    record.events_db = memory
    return evidence


def inline_requests(record: RawSimulationRecord, steps: list[int], protocol: ComparisonProtocol):
    """Stream detached wire states; never retain all reconstructed graphs."""
    batch = []
    for step, graph in record.iter_graphs(steps):
        # to_wire detaches adjacency before iter_graphs advances its graph.
        batch.append(FrozenSimulationState(step, graph, record.opinions[step], {}).to_wire())
        if len(batch) == protocol.inline_batch_size:
            yield {"dynamics_type": "HK", "measurements": ["pathway"],
                   "epsilon": record.metadata.HKParams.Tolerance, "states": batch}
            batch = []
    if batch:
        yield {"dynamics_type": "HK", "measurements": ["pathway"],
               "epsilon": record.metadata.HKParams.Tolerance, "states": batch}


def measure(reference: PrimaryReference, protocol: ComparisonProtocol, protocol_id: str) -> ComparisonResult:
    started = time.monotonic()
    try:
        directory = Path(reference.raw_directory)
        before = source_identity(directory)
        if before.fingerprint != reference.source_fingerprint:
            raise ValueError("raw source identity differs from the primary I_w source")
        steps = [*range(0, reference.terminal_step, protocol.observation_interval), reference.terminal_step]
        points = []
        with RawSimulationRecord(str(directory.parent), reference.metadata) as record:
            if record.max_step != reference.terminal_step:
                raise ValueError("raw terminal step differs from primary statistics")
            event_reader = index_events_in_memory(record)
            snapshots = verify_graph_snapshots(record)
            requests = inline_requests(record, steps, protocol)
            for response in run_probe_jobs(requests, binary_path=protocol.probe_binary):
                measured = parse_dataclass(ProbeResponse, response)
                for state in measured.results:
                    value = state.pathway
                    if value is None:
                        raise ValueError(f"missing pathway measurement at step {state.step}")
                    values = (value.separation, value.excess_concordance, value.edge_concordance, value.pair_concordance)
                    if any(item is None or not math.isfinite(item) for item in values):
                        raise ValueError(f"nonfinite pathway measurement at step {state.step}")
                    points.append(Observation(state.step, *values))
        if [point.step for point in points] != steps:
            raise ValueError("probe did not return the exact requested observation grid")
        passage = first_passage((point.step, point.P, point.E) for point in points)
        if source_identity(directory) != before:
            raise ValueError("raw source changed during P/E measurement")
        return ComparisonResult(1, "complete", protocol_id, reference, tuple(points), passage.label,
                                passage.separation_step, passage.segregation_step, time.monotonic() - started,
                                verified_graph_snapshots=snapshots, event_reader=event_reader)
    except Exception as error:
        return ComparisonResult(1, "failed", protocol_id, reference, (), None, None, None,
                                time.monotonic() - started, f"{type(error).__name__}: {error}")


def analyze(database, analysis_id, selection, output_db, probe, *, workers=2, limit=None):
    database, selection, output_db, probe = (Path(path).resolve() for path in (database, selection, output_db, probe))
    if database == output_db:
        raise ValueError("comparison output must be separate from the read-only primary database")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    tasks = load_selection(selection)
    references = load_primary(database, analysis_id, tasks)
    protocol = make_protocol(probe)
    protocol_id = fingerprint(protocol)
    plan_id = fingerprint((analysis_id, tasks, protocol_id))
    started = time.monotonic()
    completed = failed = cached = 0
    with DocumentStore(output_db) as store:
        store.put(PLAN_NAMESPACE, plan_id, {
            "schema": 1, "primary_database": str(database), "analysis_id": analysis_id,
            "selection": str(selection), "selection_sha256": sha(selection), "tasks": tasks,
            "protocol_id": protocol_id, "protocol": protocol,
        })
        jobs = []
        for reference in references:
            previous = store.get(NAMESPACE, result_key(protocol_id, reference))
            if previous is not None and previous["state"] == "complete":
                if source_identity(reference.raw_directory).fingerprint != reference.source_fingerprint:
                    raise ValueError(f"raw source changed since cached result: {reference.identity}")
                cached += 1
            else:
                jobs.append(reference)
        selected = jobs if limit is None else jobs[:limit]
        print(canonical({"event": "plan", "plan_id": plan_id, "protocol_id": protocol_id,
                         "planned": len(references), "cached": cached, "selected": len(selected)}), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(measure, reference, protocol, protocol_id): reference for reference in selected}
            for future in as_completed(futures):
                result = future.result()
                store.put(NAMESPACE, result_key(protocol_id, result.primary), result)
                completed += result.state == "complete"
                failed += result.state == "failed"
                print(canonical({"event": result.state, "origin": result.primary.identity.origin,
                                 "name": result.primary.identity.name, "observations": len(result.observations),
                                 "seconds": round(result.elapsed_seconds, 3), "error": result.error,
                                 "completed": completed, "cached": cached, "failed": failed}), flush=True)
        summary = {"plan_id": plan_id, "protocol_id": protocol_id, "planned": len(references),
                   "completed": completed + cached, "cached": cached, "failed": failed,
                   "pending": len(references) - completed - cached - failed,
                   "elapsed_seconds": time.monotonic() - started, "output_database": str(output_db)}
        store.put(PLAN_NAMESPACE, plan_id + ":status", summary)
    print(canonical(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Existing primary statistics, opened read-only")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Bound newly measured tasks; the full selection remains planned")
    args = parser.parse_args()
    result = analyze(args.database, args.analysis_id, args.selection, args.output_db, args.probe,
                     workers=args.workers, limit=args.limit)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
