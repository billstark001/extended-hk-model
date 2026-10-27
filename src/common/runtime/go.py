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
from common.runtime.failures import ProtocolError
from common.types import Backend, BuildInputs, BuildManifest, FileHashes


def restore_adapter(destination, source, *, binary_name, expected_inputs, expected_sha256):
    """Restore a measured executable and module manifests without invoking Go.

    Source selection and calibration ownership remain caller policy. This service
    authenticates the standard BuildManifest and preserves immutable provenance.
    """
    if not binary_name or Path(binary_name).name != binary_name or binary_name in (".", ".."):
        raise ValueError("binary_name must be a filename")
    destination, source = Path(destination), Path(source)
    manifest = cast(BuildManifest, json.loads((source / "manifest.json").read_text()))
    if any(manifest.get(key) != value for key, value in expected_inputs.items()):
        raise ProtocolError("calibrated runtime source/backend changed")
    if manifest["binary_sha256"] != expected_sha256:
        raise ProtocolError("calibrated adapter source/binary changed")
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in (
        (binary_name, expected_sha256),
        ("go.mod", manifest["module_sha256"]),
        ("go.sum", manifest["sum_sha256"]),
    ):
        original, target = source / name, destination / name
        if sha(original) != digest or (target.exists() and sha(target) != digest):
            raise ProtocolError("calibrated executable/module hash changed")
        if not target.exists():
            shutil.copy2(original, target)
        if sha(target) != digest:
            raise ProtocolError("calibrated executable/module copy differs")
    binary = str(destination / binary_name)
    restored = {**manifest, "binary": binary, "cached_binary": binary}
    path = destination / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != restored:
            raise ProtocolError("restored executable manifest changed")
    else:
        atomic_json(path, restored)
    return restored


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
    binary_name: str = "paper-theory-engine",
) -> BuildManifest:
    """Build the shared adapter against both supplied runtime source checkouts.

    Generate local module replacements and caches inside destination; preserve
    runtime checkout files. Return the source/toolchain/backend/binary manifest.
    A matching snapshot is reused; different inputs or a changed binary require
    a new destination. Source edits during compilation are rejected.
    """
    if not binary_name or Path(binary_name).name != binary_name or binary_name in (".", ".."):
        raise ValueError("binary_name must be a filename")
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
    binary = destination / binary_name
    manifest = destination / "manifest.json"
    if binary.exists() and manifest.exists():
        old = cast(BuildManifest, json.loads(manifest.read_text()))
        if old["source_fingerprint"] == ident and old["binary_sha256"] == sha(binary):
            # The scientific manifest stays immutable; a caller may recreate a
            # removed APFS execution stage from its verified provenance copy.
            if link_stage is not None:
                target = link_stage / binary_name
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
    target = link_stage / binary_name if link_stage is not None else binary
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
