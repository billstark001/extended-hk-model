"""Resume one frozen run, retaining its exact scientific executables and plan."""

import os
import shutil
from pathlib import Path
from typing import cast

from common.experiments.lifecycle import read_owned
from common.io.json import atomic_json, sha
from common.runtime.go import engine_manifest, source_manifest
from theory.runtime import measure as schema

from ..types import RunManifest
from .design import PROFILES, canonical_profile


def settings(work: Path, publish: Path, args) -> RunManifest:
    """Read the original run manifest; explicit overrides must agree."""
    manifest = work / "run_manifest.json"
    old = None
    if args.resume:
        old = cast(RunManifest, read_owned(manifest, work))
        if canonical_profile(old["profile"]) not in PROFILES:
            raise ValueError("resume requires a canonical main task profile")
        for name in ("profile", "backend", "horizon"):
            explicit = getattr(args, name)
            saved = old[name]
            agrees = (
                canonical_profile(explicit) == canonical_profile(str(saved))
                if name == "profile" and explicit is not None
                else explicit == saved
            )
            if explicit is not None and not agrees:
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
    for binding in (work / "evidence").glob("*/*/binding.json"):
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
        bindings = list((work / "evidence").glob("*/*/binding.json"))
        raw = [
            p
            for p in (work / "evidence").glob("*/*/raw/*")
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
