"""Explicit migration of frozen job output protocols; scientific state is untouched."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha
from common.storage.documents import DocumentWriter
from common.experiments.lifecycle import session_lock
from ehk.domain import PersistentJob, PolicyChange, ScenarioMetadata, WireRecord, parse_dataclass, to_wire


@dataclass(kw_only=True, slots=True)
class RetiredJob(WireRecord):
    base_path: str
    metadata: ScenarioMetadata
    observe_at: tuple[int, ...]
    summary_path: str
    fixed_horizon: bool = True
    policy_change: PolicyChange | None = None
    exposure_trace: str | None = None
    trace_start: int | None = None


@dataclass(frozen=True, slots=True)
class JobCandidate:
    path: Path
    original: bytes
    digest: str
    migrated: PersistentJob


@dataclass(frozen=True, slots=True)
class JobChange:
    path: str
    old_sha256: str
    new_fingerprint: str
    backup: str


@dataclass(frozen=True, kw_only=True, slots=True)
class JobMigrationReceipt:
    schema: int
    jobs: tuple[JobChange, ...]
    binding_changed: bool = False
    scientific_state_changed: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class JobMigrationState(JobMigrationReceipt):
    state: str


def migrate_jobs(directory) -> JobMigrationReceipt:
    directory = Path(directory).resolve()
    with session_lock(directory / "runtime"):
        return _migrate_jobs(directory)


def _migrate_jobs(directory: Path) -> JobMigrationReceipt:
    candidates = []
    for path in directory.rglob("*.json"):
        if path.name.startswith("._") or (path.name != "job.json" and path.parent.name != ".jobs"):
            continue
        original = path.read_bytes()
        wire = json.loads(original)
        if "summary_path" not in wire:
            continue
        if "result_protocol" in wire:
            raise ValueError(f"job mixes file and stream protocols: {path}")
        retired = parse_dataclass(RetiredJob, wire)
        migrated = PersistentJob(base_path=retired.base_path, metadata=retired.metadata,
            observe_at=retired.observe_at, fixed_horizon=retired.fixed_horizon,
            policy_change=retired.policy_change, exposure_trace=retired.exposure_trace,
            trace_start=retired.trace_start, result_protocol="smp-research-result-v1",
            _wire_fields=frozenset((retired._wire_fields - {"summary_path"}) | {"result_protocol"}))
        candidates.append(JobCandidate(path, original, sha(path), migrated))
    history = directory / "runtime/job_protocol_history"
    changes = []
    for candidate in candidates:
        if sha(candidate.path) != candidate.digest:
            raise ValueError(f"frozen job changed during migration: {candidate.path}")
        backup = history / f"{candidate.digest}.json"
        if backup.exists() and backup.read_bytes() != candidate.original:
            raise ValueError("job history conflicts with original bytes")
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(candidate.original)
        changes.append(JobChange(candidate.path.relative_to(directory).as_posix(), candidate.digest,
            fingerprint(candidate.migrated), backup.relative_to(directory).as_posix()))
    receipt = JobMigrationReceipt(schema=1, jobs=tuple(changes))
    # Authenticate the intent before replacing only the job control metadata.
    identity = fingerprint(receipt)
    with DocumentWriter(directory / "analysis.sqlite") as store:
        store.put("ehk-job-protocol-migrations-v1", identity, JobMigrationState(schema=receipt.schema, jobs=receipt.jobs, binding_changed=receipt.binding_changed, scientific_state_changed=receipt.scientific_state_changed, state="prepared"))
        for candidate in candidates:
            atomic_json(candidate.path, candidate.migrated)
        store.put("ehk-job-protocol-migrations-v1", identity, JobMigrationState(schema=receipt.schema, jobs=receipt.jobs, binding_changed=receipt.binding_changed, scientific_state_changed=receipt.scientific_state_changed, state="complete"))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(to_wire(migrate_jobs(args.directory)), indent=2))


if __name__ == "__main__":
    main()
