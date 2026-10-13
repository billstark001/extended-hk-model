"""Build against the actual Go solvers and record source hashes; no source copies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

from common.io.json import atomic_json, fingerprint, sha
from common.runtime.environment import runtime_environment
from common.types import Backend, BuildInputs, BuildManifest, FileHashes


def source_manifest(root: Path) -> FileHashes:
    """Hash production Go sources plus go.mod/go.sum in a runtime working tree; exclude tests, vendor and binary directories."""
    sources: list[Path] = []
    excluded = {".git", "bin", "vendor", "outputs", "inputs", "artifacts", "datasets", ".venv", ".cache"}
    for directory, children, files in os.walk(root):
        children[:] = [name for name in children if name not in excluded]
        sources.extend(
            Path(directory) / name
            for name in files
            if name.endswith(".go") and not name.endswith("_test.go") and not name.startswith("._")
        )
    return {str(p.relative_to(root)): sha(p) for p in sorted(sources)} | {
        name: sha(root / name) for name in ("go.mod", "go.sum")
    }


def engine_manifest(engine_root: Path) -> FileHashes:
    """Hash the custom adapter production Go files and declared go.mod, excluding benchmark/test files."""
    return {p.name: sha(p) for p in sorted(engine_root.glob("*.go")) if not p.name.endswith("_test.go")} | {
        "go.mod": sha(engine_root / "go.mod")
    }


def build_adapter(
    destination: Path,
    meso_root: Path,
    micro_root: Path,
    backend: Backend,
    engine_root: Path,
    *,
    link_stage: Path | None = None,
) -> BuildManifest:
    """Build the shared adapter against both supplied runtime source checkouts.

    Generate local module replacements and caches inside destination; preserve
    runtime checkout files. Return the source/toolchain/backend/binary manifest.
    A matching snapshot is reused; different inputs or a changed binary require
    a new destination. Source edits during compilation are rejected.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    env = runtime_environment(destination)
    if link_stage is not None:
        link_stage = Path(link_stage).resolve()
        (link_stage / "tmp").mkdir(parents=True, exist_ok=True)
        env["GOTMPDIR"] = str(link_stage / "tmp")
        # The external linker uses TMPDIR independently of Go's work directory.
        env["TMPDIR"] = str(link_stage / "tmp")
    inputs: BuildInputs = {
        "meso": source_manifest(meso_root),
        "micro": source_manifest(micro_root),
        "engine": engine_manifest(engine_root),
        "backend": backend,
        "go_version": subprocess.check_output(["go", "version"], text=True).strip(),
    }
    ident = fingerprint(inputs)
    binary = destination / "paper-theory-engine"
    manifest = destination / "manifest.json"
    if binary.exists() and manifest.exists():
        old = cast(BuildManifest, json.loads(manifest.read_text()))
        if old["source_fingerprint"] == ident and old["binary_sha256"] == sha(binary):
            # The scientific manifest stays immutable; a caller may recreate a
            # removed APFS execution stage from its verified provenance copy.
            if link_stage is not None:
                target = link_stage / "paper-theory-engine"
                shutil.copy2(binary, target)
                if sha(target) != old["binary_sha256"]:
                    raise RuntimeError("staged binary copy differs")
                return {**old, "binary": str(target), "cached_binary": str(binary)}
            return {**old, "binary": str(binary), "cached_binary": str(binary)}
        raise RuntimeError("source/backend changed: use a NEW build directory; existing run snapshots are immutable")
    (destination / "go.mod").write_text(
        (engine_root / "go.mod").read_text()
        + "\nreplace smp => "
        + json.dumps(str(micro_root.resolve()))
        + "\nreplace smp-meso => "
        + json.dumps(str(meso_root.resolve()))
        + "\n"
    )
    (destination / "go.sum").write_text((micro_root / "go.sum").read_text() + (meso_root / "go.sum").read_text())
    subprocess.run(["go", "mod", "tidy", "-modfile", str(destination / "go.mod")], cwd=engine_root, env=env, check=True)
    target = link_stage / "paper-theory-engine" if link_stage is not None else binary
    command = ["go", "build", "-modfile", str(destination / "go.mod"), "-p", "4", "-o", str(target)]
    if backend != "purego":
        command += ["-tags", backend]
    subprocess.run(command + ["."], cwd=engine_root, env=env, check=True)
    if target != binary:
        shutil.copy2(target, binary)
        if sha(target) != sha(binary):
            raise RuntimeError("staged binary copy differs")
    if (
        source_manifest(meso_root) != inputs["meso"]
        or source_manifest(micro_root) != inputs["micro"]
        or engine_manifest(engine_root) != inputs["engine"]
    ):
        raise RuntimeError("source changed during build; use a new directory and rebuild")
    result: BuildManifest = {
        **inputs,
        "source_fingerprint": ident,
        "binary_sha256": sha(binary),
        "binary": str(target),
        "cached_binary": str(binary),
        "meso_git_head": subprocess.check_output(["git", "-C", str(meso_root), "rev-parse", "HEAD"], text=True).strip(),
        "micro_git_head": subprocess.check_output(
            ["git", "-C", str(micro_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_sha256": sha(destination / "go.mod"),
        "sum_sha256": sha(destination / "go.sum"),
    }
    atomic_json(manifest, result)
    return result
