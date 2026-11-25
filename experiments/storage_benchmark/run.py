"""Prepare and execute a sequential, isolated storage benchmark campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha

from . import corpus
from .common import SCALARS, document_digest, header_document, payload_size
from .corpus import PROFILES, get_profile, iter_documents, load_sources

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("current", "tinymongo", "indexed_split")
BACKEND_CHOICES = (*BACKENDS, "sqlite_optimized")
ACTIONS = (
    "point_metadata",
    "point_full",
    "selective_metadata",
    "scan_metadata",
    "scan_full",
    "nested_metadata",
    "state_update",
    "replace_full",
)
ACTION_CHOICES = (*ACTIONS[:-2], "scope_metadata", "scope_full", *ACTIONS[-2:])


def _selection(values, choices, label):
    if isinstance(values, str):
        raise ValueError(f"{label} must be a nonempty sequence of names")
    try:
        selected = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be a nonempty sequence of names") from error
    if not selected:
        raise ValueError(f"{label} must contain at least one name")
    if any(name not in choices for name in selected):
        raise ValueError(f"unknown {label}; choices are {', '.join(choices)}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate {label}")
    return selected


def _environment(vendor):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(vendor), str(ROOT / "src"), str(ROOT)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def _hardware():
    values = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
    }
    if sys.platform == "darwin":
        for field, name in (("ram_bytes", "hw.memsize"), ("cpu", "machdep.cpu.brand_string")):
            values[field] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    with sqlite3.connect(":memory:") as connection:
        values["sqlite_compile_options"] = [row[0] for row in connection.execute("PRAGMA compile_options")]
    return values


def _sources_pack(directory):
    sources = dict(load_sources())
    sources["lengths"] = sources["lengths"].tolist()
    sources["length_probabilities"] = sources["length_probabilities"].tolist()
    sources["actual_files"] = [str(path) for path in sources["actual_files"]]
    path = directory / "sources-pack.json"
    atomic_json(path, sources)
    return path


def _unpack_sources(path):
    """Restore exactly the source types restored by the measured worker."""
    import numpy as np

    sources = json.loads(Path(path).read_text())
    sources["lengths"] = np.asarray(sources["lengths"], dtype=np.int64)
    sources["length_probabilities"] = np.asarray(sources["length_probabilities"], dtype=float)
    sources["actual_files"] = [Path(value) for value in sources["actual_files"]]
    return sources


def _preflight_indices(profile, sources):
    """Cover families, every mechanism task, and every revision before loading."""
    if profile == "actual_available":
        count = len(sources["actual_files"])
        return sorted({0, count // 2, count - 1}) if count else []
    if profile not in ("full_paper", "versioned_2x"):
        raise ValueError(f"unknown corpus profile: {profile}")
    tasks = sources["tasks"]
    family_indices = {}
    for index, task in enumerate(tasks):
        family_indices.setdefault(task["mode"], []).append(index)
    representatives = {0, len(tasks) // 2, len(tasks) - 1}
    for family, indices in family_indices.items():
        representatives.update(indices if family == "mech" else (indices[0], indices[-1]))
    revisions = 2 if profile == "versioned_2x" else 1
    selected = {revision * len(tasks) + index for revision in range(revisions) for index in representatives}
    for revision in range(1, revisions):
        selected.update(revision * len(tasks) + offset for offset in (-1, 0, 1))
    return sorted(index for index in selected if 0 <= index < revisions * len(tasks))


def _preflight_source_pack(directory, source_pack, profiles):
    """Compare live and JSON-roundtripped documents before any catalog or DB load."""
    live_sources = load_sources()
    packed_sources = _unpack_sources(source_pack)
    original_loader = corpus.load_sources
    report = {
        "schema": 1,
        "status": "running",
        "sources_pack_sha256": sha(source_pack),
        "sampling": "every family first/last, all mechanism tasks, every revision and adjacent revision boundaries; independent of --limit",
        "profiles": {},
    }
    path = Path(directory) / "source-pack-preflight.json"
    try:
        for profile in profiles:
            indices = _preflight_indices(profile, live_sources)
            checked = []
            for index in indices:
                corpus.load_sources = lambda: live_sources
                expected = corpus.document_at(index, profile)
                expected_full = document_digest(expected)
                expected_header = document_digest(header_document(expected))
                family, revision = expected.get("mode"), expected.get("analysis_id")
                del expected
                corpus.load_sources = lambda: packed_sources
                actual = corpus.document_at(index, profile)
                actual_full = document_digest(actual)
                actual_header = document_digest(header_document(actual))
                del actual
                if (actual_full, actual_header) != (expected_full, expected_header):
                    raise ValueError(
                        f"source-pack roundtrip changed corpus document: {profile} index={index} family={family}"
                    )
                checked.append(
                    {
                        "index": index,
                        "family": family,
                        "analysis_id": revision,
                        "full_digest": expected_full,
                        "header_digest": expected_header,
                    }
                )
            report["profiles"][profile] = checked
        report["status"] = "passed"
        report["checked_documents"] = sum(len(values) for values in report["profiles"].values())
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        corpus.load_sources = original_loader
        atomic_json(path, report)
    return path, report


def _observe_environment(directory, phase, identifier):
    """Record command names rather than arguments; observations are not timed work."""
    observation = {"time_unix": time.time(), "phase": phase, "job": identifier}
    try:
        process = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pcpu=,rss=,comm="], text=True, capture_output=True, check=True, timeout=5
        )
        processes = []
        for line in process.stdout.splitlines():
            values = line.split(None, 4)
            if len(values) != 5:
                continue
            pid, ppid, cpu, rss, command = values
            processes.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "cpu_percent": float(cpu),
                    "rss_kib": int(rss),
                    "command": Path(command).name,
                }
            )
        observation["processes"] = sorted(processes, key=lambda item: item["cpu_percent"], reverse=True)[:12]
        observation["cpu_measurement"] = "ps pcpu; not an isolated CPU allocation or instantaneous utilization sample"
    except Exception as error:
        observation.update(unavailable=True, error_type=type(error).__name__, error=str(error))
    try:
        with (Path(directory) / "environment-observations.jsonl").open("a") as output:
            output.write(json.dumps(observation, sort_keys=True) + "\n")
    except OSError:
        pass  # Best-effort environment diagnostics must not change benchmark success.


def prepare(
    directory,
    vendor,
    *,
    profiles=PROFILES,
    backends=BACKENDS,
    actions=ACTIONS,
    repetitions=3,
    query_repetitions=2,
    limit=None,
):
    backends = _selection(backends, BACKEND_CHOICES, "backends")
    selected_actions = _selection(actions, ACTION_CHOICES, "actions")
    # Reads use the original catalog; retain the protocol's mutation order.
    actions = tuple(action for action in ACTION_CHOICES if action in selected_actions)
    directory, vendor = Path(directory).resolve(), Path(vendor).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "plan.json").exists():
        raise ValueError("benchmark directory is already frozen; use run to resume or choose a new directory")
    if repetitions < 1 or query_repetitions < 1 or limit is not None and limit < 1:
        raise ValueError("repetitions and limits must be positive")
    source_pack = _sources_pack(directory)
    preflight_path, preflight = _preflight_source_pack(directory, source_pack, profiles)
    profile_info = {}
    for profile in profiles:
        info = get_profile(profile)
        count = min(limit, info["count"]) if limit is not None else info["count"]
        path = directory / f"catalog-{profile}.jsonl"
        raw_bytes = 0
        combined = hashlib.sha256()
        with path.open("w") as output:
            for index, document in enumerate(iter_documents(profile, limit=count)):
                full_digest = document_digest(document)
                row = {
                    "_id": document["_id"],
                    "index": index,
                    **{key: document.get(key) for key in SCALARS},
                    "metadata": {"HKParams": document.get("metadata", {}).get("HKParams", {})},
                    "full_digest": full_digest,
                    "header_digest": document_digest(header_document(document)),
                }
                combined.update(full_digest.encode())
                raw_bytes += payload_size(document["payload"])
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        profile_info[profile] = {
            **info,
            "measured_count": count,
            "catalog": str(path),
            "catalog_sha256": sha(path),
            "payload_logical_bytes": raw_bytes,
            "corpus_digest": combined.hexdigest(),
        }
        print(json.dumps({"prepared": profile, "count": count, "payload_bytes": raw_bytes}), flush=True)
    source_files = {str(path.relative_to(ROOT)): sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    source_files["src/common/storage/documents.py"] = sha(ROOT / "src/common/storage/documents.py")
    package_report = directory.parent / "install-report.json"
    candidates = [vendor.parent / "install-report.json", package_report]
    installed = next((str(path) for path in candidates if path.is_file()), None)
    plan = {
        "schema": 2,
        "study": "isolated_storage_benchmark",
        "profiles": profile_info,
        "backends": list(backends),
        "repetitions": repetitions,
        "query_repetitions": query_repetitions,
        "actions": list(actions),
        "vendor": str(vendor),
        "sources_pack": str(source_pack),
        "sources_pack_sha256": sha(source_pack),
        "source_pack_preflight": {
            "path": str(preflight_path),
            "sha256": sha(preflight_path),
            "checked_documents": preflight["checked_documents"],
        },
        "source_files": source_files,
        "hardware": _hardware(),
        "package_install_report": installed,
        "durability": {"journal_mode": "wal", "synchronous": "FULL"},
        "timing_protocol": {
            "processes": "one measured process at a time; fresh process for each action/repetition",
            "backend_order": "cyclically rotated over profiles and repetitions",
            "page_cache": "OS page cache uncontrolled; no claim of disk-cold reads",
            "load": "per-document serialization+upsert+commit; source generation/hash excluded from operation timer",
            "load_end_to_end": "includes streaming generation and corpus content checks",
            "reads": "backend call and materialization timed; exact value/hash verification outside timer",
            "memory": "whole-process ru_maxrss plus 10ms RSS sampling over action window; baseline retained",
            "recovery": "separate SIGKILL campaign, not timed and not a power-cut test",
            "environment": "best-effort ps top CPU command names before/after each job; no process arguments or CPU isolation",
        },
    }
    recovery = directory.parent / "recovery_final"
    if recovery.exists():
        plan["recovery"] = {name: str(recovery / name / "recovery.json") for name in backends}
    plan["fingerprint"] = fingerprint(plan)
    atomic_json(directory / "plan.json", plan)
    return plan


def _execute(request, result_path, directory, environment, *, timeout=900):
    import psutil

    request_path = directory / "requests" / (result_path.stem + ".json")
    atomic_json(request_path, request)
    log_path = directory / "logs" / (result_path.stem + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.storage_benchmark.worker",
        str(request_path),
        "--output",
        str(result_path),
    ]
    peak = 0
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > 4 * 1024**3:
                    raise RuntimeError("benchmark process exceeded 4 GiB RSS limit")
                if time.monotonic() - started > timeout:
                    raise TimeoutError("benchmark action exceeded bounded runtime")
                time.sleep(0.025)
            if process.returncode != 0:
                raise RuntimeError(f"benchmark worker failed ({process.returncode}); see {log_path}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    result = json.loads(result_path.read_text())
    result["parent_monitored_peak_rss_bytes"] = peak
    result["worker_wall_seconds"] = time.monotonic() - started
    result["request_fingerprint"] = fingerprint(request)
    atomic_json(result_path, result)
    return result


def run(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("study") != "isolated_storage_benchmark":
        raise ValueError("not an isolated benchmark directory")
    frozen = dict(plan)
    recorded_fingerprint = frozen.pop("fingerprint")
    if fingerprint(frozen) != recorded_fingerprint:
        raise ValueError("frozen benchmark plan changed; use a new campaign")
    for relative, expected in plan["source_files"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"frozen benchmark source changed: {relative}; use a new campaign")
    for info in plan["profiles"].values():
        if sha(info["catalog"]) != info["catalog_sha256"]:
            raise ValueError("frozen benchmark catalog changed; use a new campaign")
    if sha(plan["sources_pack"]) != plan["sources_pack_sha256"]:
        raise ValueError("frozen benchmark source pack changed; use a new campaign")
    preflight = plan.get("source_pack_preflight")
    if plan.get("schema", 1) >= 2 and preflight is None:
        raise ValueError("source-pack preflight is missing; use a new campaign")
    if preflight is not None:
        if sha(preflight["path"]) != preflight["sha256"]:
            raise ValueError("frozen source-pack preflight changed; use a new campaign")
        evidence = json.loads(Path(preflight["path"]).read_text())
        if evidence.get("status") != "passed" or evidence.get("sources_pack_sha256") != plan["sources_pack_sha256"]:
            raise ValueError("source-pack preflight did not pass for the frozen sources")
    backends = _selection(plan.get("backends"), BACKEND_CHOICES, "backends")
    _selection(plan.get("actions"), ACTION_CHOICES, "actions")
    environment = _environment(plan["vendor"])
    if str(plan["vendor"]) not in sys.path:
        sys.path.insert(0, plan["vendor"])
    jobs = []
    for profile_index, (profile, info) in enumerate(plan["profiles"].items()):
        for action in ("load", *plan["actions"]):
            for repeat in range(plan["repetitions"]):
                offset = (repeat + profile_index) % len(backends)
                order = (*backends[offset:], *backends[:offset])
                for backend in order:
                    identifier = f"{profile}--{backend}--{action}--{repeat}"
                    result_path = directory / "results" / (identifier + ".json")
                    request = {
                        "profile": profile,
                        "backend": backend,
                        "action": action,
                        "repeat": repeat,
                        "directory": str(directory / "databases" / profile / backend / str(repeat)),
                        "catalog": info["catalog"],
                        "limit": info["measured_count"],
                        "sources_pack": plan["sources_pack"],
                        "sources_pack_sha256": plan["sources_pack_sha256"],
                        "query_repetitions": plan["query_repetitions"],
                        "point_operations": 200,
                        "update_operations": 200,
                        "replace_operations": 20,
                    }
                    jobs.append((identifier, request, result_path))
    return _run_jobs(directory, jobs, environment)


def _run_jobs(directory, jobs, environment):
    completed = []
    active_identifier = None
    try:
        for number, (identifier, request, result_path) in enumerate(jobs, 1):
            active_identifier = identifier
            if result_path.is_file():
                saved = json.loads(result_path.read_text())
                if saved.get("request_fingerprint") == fingerprint(request) and saved.get("status") in (
                    "ok",
                    "not_applicable",
                ):
                    completed.append(identifier)
                    continue
                raise ValueError(f"incomplete existing result: {result_path}; review before restarting")
            attempt_path = directory / "requests" / (result_path.stem + ".json")
            if attempt_path.exists():
                raise ValueError(
                    f"incomplete previous attempt: {attempt_path}; use a new campaign to avoid timing a partly mutated database"
                )
            if request["action"] == "load" and Path(request["directory"]).exists():
                raise ValueError("load database already exists; use a new campaign for clean insertion timings")
            if shutil.disk_usage(directory).free < 8 * 1024**3:
                raise RuntimeError("less than 8 GiB free; benchmark stopped without touching production data")
            print(json.dumps({"job": number, "planned": len(jobs), "running": identifier}), flush=True)
            atomic_json(
                directory / "status.json",
                {"state": "running", "completed": len(completed), "planned": len(jobs), "active": identifier},
            )
            _observe_environment(directory, "before", identifier)
            try:
                result = _execute(request, result_path, directory, environment)
            except BaseException:
                _observe_environment(directory, "after_failed", identifier)
                raise
            _observe_environment(directory, "after", identifier)
            completed.append(identifier)
            atomic_json(
                directory / "status.json",
                {"state": "running", "completed": len(completed), "planned": len(jobs), "last": identifier},
            )
            print(
                json.dumps(
                    {
                        "finished": identifier,
                        "status": result["status"],
                        "seconds": result.get("latency", {}).get("seconds"),
                        "rss_mib": result.get("peak_rss_action_bytes", 0) / 1024**2,
                    }
                ),
                flush=True,
            )
    except BaseException as error:
        atomic_json(
            directory / "status.json",
            {
                "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "completed": len(completed),
                "planned": len(jobs),
                "failed_job": active_identifier,
                "error_type": type(error).__name__,
                "error": str(error),
                "time_unix": time.time(),
            },
        )
        raise
    atomic_json(directory / "status.json", {"state": "complete", "completed": len(completed), "planned": len(jobs)})
    return {"completed": len(completed), "planned": len(jobs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report", "all"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, default=ROOT / "outputs/benchmarks/storage_20261122/vendor")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    parser.add_argument("--backends", nargs="+", choices=BACKEND_CHOICES, help="Backends to freeze during prepare/all.")
    parser.add_argument(
        "--actions", nargs="+", choices=ACTION_CHOICES, help="Actions to freeze during prepare/all; load is always run."
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--query-repetitions", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.action not in ("prepare", "all") and (args.backends is not None or args.actions is not None):
        parser.error("--backends and --actions belong to prepare/all; run/report use the frozen plan")
    if args.action in ("prepare", "all"):
        prepare(
            args.directory,
            args.vendor,
            profiles=args.profiles,
            backends=BACKENDS if args.backends is None else args.backends,
            actions=ACTIONS if args.actions is None else args.actions,
            repetitions=args.repetitions,
            query_repetitions=args.query_repetitions,
            limit=args.limit,
        )
    if args.action in ("run", "all"):
        run(args.directory)
    if args.action in ("report", "all"):
        from .report import render

        render(args.directory)


if __name__ == "__main__":
    main()
