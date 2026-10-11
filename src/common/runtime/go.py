"""Build against the actual Go solvers and record source hashes; no source copies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha
from common.runtime.environment import runtime_environment


def source_manifest(root):
    """Hash production Go sources plus go.mod/go.sum in a runtime working tree; exclude tests, vendor and binary directories."""
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*.go"))
        if not p.name.endswith("_test.go") and not any(x in p.parts for x in (".git", "bin", "vendor"))
    } | {p: sha(root / p) for p in ("go.mod", "go.sum")}


def engine_manifest(engine_root):
    """Hash the custom adapter production Go files and declared go.mod, excluding benchmark/test files."""
    return {p.name: sha(p) for p in sorted(engine_root.glob("*.go")) if not p.name.endswith("_test.go")} | {
        "go.mod": sha(engine_root / "go.mod")
    }


def build_adapter(destination, meso_root, micro_root, backend, engine_root):
    """Build the shared adapter against both supplied runtime source checkouts.

    Generate local module replacements and caches inside destination; preserve
    runtime checkout files. Return the source/toolchain/backend/binary manifest.
    A matching snapshot is reused; different inputs or a changed binary require
    a new destination. Source edits during compilation are rejected.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    env = runtime_environment(destination)
    inputs = {
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
        old = json.loads(manifest.read_text())
        if old["source_fingerprint"] == ident and old["binary_sha256"] == sha(binary):
            return old
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
    command = ["go", "build", "-modfile", str(destination / "go.mod"), "-p", "4", "-o", str(binary)]
    if backend != "purego":
        command += ["-tags", backend]
    subprocess.run(command + ["."], cwd=engine_root, env=env, check=True)
    if (
        source_manifest(meso_root) != inputs["meso"]
        or source_manifest(micro_root) != inputs["micro"]
        or engine_manifest(engine_root) != inputs["engine"]
    ):
        raise RuntimeError("source changed during build; use a new directory and rebuild")
    result = {
        **inputs,
        "source_fingerprint": ident,
        "binary_sha256": sha(binary),
        "binary": str(binary),
        "meso_git_head": subprocess.check_output(["git", "-C", str(meso_root), "rev-parse", "HEAD"], text=True).strip(),
        "micro_git_head": subprocess.check_output(
            ["git", "-C", str(micro_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_sha256": sha(destination / "go.mod"),
        "sum_sha256": sha(destination / "go.sum"),
    }
    atomic_json(manifest, result)
    return result
