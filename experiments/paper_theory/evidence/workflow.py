"""Select, freeze and execute the new evidence design under one output root."""

import json
from collections import Counter
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha
from common.runtime.process import acquire_lock, release_lock
from theory.runtime.store import write_design

from .design import design


def plan(directory, profile, horizon=None):
    candidate = design(profile, horizon)
    path = directory / "plan.json"
    candidate = {**candidate, "plan_fingerprint": fingerprint(candidate)}
    if path.exists():
        if json.loads(path.read_text()) != candidate:
            raise ValueError("evidence plan changed; use a new output directory")
    else:
        atomic_json(path, candidate)
    for kind in ("measure", "micro"):
        write_design(directory / kind / "design.json", [t for t in candidate["tasks"] if t["kind"] == kind])
    print(dict(Counter(t["kind"] for t in candidate["tasks"])), flush=True)
    return candidate


def perform(a, root, build_dir, meso, micro):
    """Dispatch evidence stages; source builds and scientific execution stay separate."""
    directory = root / "evidence" / a.profile
    if a.action == "build":
        from smp_meso_bindings import build_binary

        from ..runtime.build import build

        manifest = build(build_dir, meso, micro, a.backend)
        binary = a.binary or root / "bin" / "smp-kinetic"
        build_binary(command_name="kinetic", source_root=meso, output_path=binary, backend=a.backend)
        # Bind the public binary to actual production source hashes too.
        from common.runtime.go import source_manifest

        atomic_json(
            root / "bin" / "kinetic_manifest.json",
            {
                "binary": str(binary),
                "binary_sha256": sha(binary),
                "source": source_manifest(meso),
                "backend": a.backend,
                "adapter_source": manifest["engine"],
            },
        )
        return
    if a.action in ("plan", "run", "smoke"):
        candidate = plan(directory, a.profile, a.horizon)
        if a.action == "plan":
            return
        from ..micro_validation.population.run import execute as population
        from ..runtime.measure_runs import execute as measure

        binary = a.binary or root / "bin" / "smp-kinetic"
        manifest = json.loads((build_dir / "manifest.json").read_text())
        engine = Path(manifest["binary"])
        if sha(engine) != manifest["binary_sha256"]:
            raise ValueError("engine binary differs from build manifest")
        lock = directory / "active.lock"
        acquire_lock(lock)
        try:
            measure(
                [t for t in candidate["tasks"] if t["kind"] == "measure"],
                directory / "measure",
                binary,
                force=a.force,
                limit=a.limit,
            )
            population(
                [t for t in candidate["tasks"] if t["kind"] == "micro"],
                directory / "micro" / "raw",
                engine,
                force=a.force,
                limit=a.limit,
            )
        finally:
            release_lock(lock)
        return
    if a.action == "analyze":
        from .analyze import analyze

        return analyze(directory)
    if a.action == "plot":
        from .plot import plot

        return plot(directory)
    if a.action == "status":
        candidate = json.loads((directory / "plan.json").read_text())
        complete = sum(
            (directory / t["kind"] / "raw" / (t["id"] + (".npz" if t["kind"] == "measure" else ".json"))).exists()
            for t in candidate["tasks"]
        )
        print({"planned": len(candidate["tasks"]), "complete": complete})
        return
    raise ValueError("action unavailable for evidence")
