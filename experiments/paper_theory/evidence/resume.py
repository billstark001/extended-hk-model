"""Resume one frozen run, retaining its exact scientific executables and plan."""

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from common.io.json import atomic_json, sha
from common.runtime.go import engine_manifest, source_manifest
from theory.runtime import measure as schema

from ..types import RunManifest


@contextmanager
def session_lock(work):
    """OS releases this lock even on SIGKILL; never unlink a flock inode."""
    with (work / "session.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"run already active: {work}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_owned(path, work):
    if not path.resolve().is_relative_to(work):
        raise ValueError(f"run metadata links outside its directory: {path}")
    return json.loads(path.read_text())


def settings(work: Path, publish: Path, args) -> RunManifest:
    """Read the original run manifest; explicit overrides must agree."""
    manifest = work / "run_manifest.json"
    old = None
    if args.resume:
        old = cast(RunManifest, read_owned(manifest, work))
        if old["profile"] not in ("main", "main_smoke"):
            raise ValueError("resume requires a canonical main or main_smoke profile")
        for name in ("profile", "backend", "horizon"):
            explicit = getattr(args, name)
            saved = old[name]
            if explicit is not None and explicit != saved:
                raise ValueError(f"resume cannot change {name}; use a new run")
            setattr(args, name, saved)
    else:
        args.profile = args.profile or "main"
        args.backend = args.backend or ("accelerate" if os.uname().sysname == "Darwin" else "purego")
    engine = Path(__file__).resolve().parents[1] / "runtime" / "engine"
    candidate: RunManifest = {
        "schema": 1,
        "work_dir": str(work),
        "publish_dir": str(publish),
        "profile": args.profile,
        "horizon": args.horizon,
        "backend": args.backend,
        "sources": {
            "meso": source_manifest(args.meso_root.resolve()),
            "micro": source_manifest(args.micro_root.resolve()),
            "engine": engine_manifest(engine),
        },
        "request_schema_sha256": sha(Path(schema.__file__)),
    }
    if old is not None:
        for key, value in old.items():
            if candidate.get(key) != value:
                raise ValueError(f"resume {key} changed; use a new run")
    # Binding files authenticate the request schema originally used.
    for binding in (work / "evidence").glob("main*/*/binding.json"):
        if read_owned(binding, work)["request_schema_sha256"] != candidate["request_schema_sha256"]:
            raise ValueError("resume request schema changed; use a new run")
    if not manifest.exists():
        atomic_json(manifest, candidate)
    return candidate


def restore_binaries(work, stage, config):
    """Copy verified Cache executables to APFS; never rebuild bound results."""
    paths = [
        (work / "build" / "manifest.json", work / "build" / "paper-theory-engine"),
        (work / "bin" / "kinetic_manifest.json", work / "bin" / "smp-kinetic"),
    ]
    if not all(manifest.exists() and binary.exists() for manifest, binary in paths):
        bindings = list((work / "evidence").glob("main*/*/binding.json"))
        raw = [
            p
            for p in (work / "evidence").glob("main*/*/raw/*")
            if not p.name.startswith("._") and (p.suffix in (".npz", ".json") or p.name.endswith(".msgpack.lz4"))
        ]
        if bindings or raw:
            raise ValueError("bound run lacks original executables; cannot safely rebuild")
        return None  # Interrupted before scientific execution; build may continue.
    restored = []
    for manifest_path, binary in paths:
        manifest = read_owned(manifest_path, work)
        if manifest["backend"] != config["backend"]:
            raise ValueError("cached backend differs from frozen run")
        expected_sources = (
            {name: manifest[name] for name in ("meso", "micro", "engine")}
            if binary.name == "paper-theory-engine"
            else {"meso": manifest["source"], "engine": manifest["adapter_source"]}
        )
        if any(config["sources"][name] != value for name, value in expected_sources.items()):
            raise ValueError("cached executable sources differ from frozen run")
        if not binary.resolve().is_relative_to(work) or sha(binary) != manifest["binary_sha256"]:
            raise ValueError(f"original executable missing or changed: {binary}")
        target = stage / binary.name
        shutil.copy2(binary, target)
        if sha(target) != manifest["binary_sha256"]:
            raise RuntimeError(f"restored executable hash differs: {target}")
        restored.append(target)
    return tuple(restored)


def recover_task_lock(directory):
    """Recover a task PID lock only when its owner has actually exited."""
    path = directory / "active.lock"
    if not path.exists():
        return
    pid = int(path.read_text())
    if pid <= 0:
        raise ValueError(f"invalid PID lock: {path}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        path.unlink()
    else:
        raise RuntimeError(f"run already active (PID {pid}): {directory}")
