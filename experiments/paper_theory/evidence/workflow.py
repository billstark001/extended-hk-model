"""Select, freeze and execute the new evidence design under one output root."""

import json
import shutil
from collections import Counter
from pathlib import Path

from common.io.json import atomic_json, fingerprint, sha
from common.runtime.failures import BatchTaskError
from common.runtime.process import acquire_lock, release_lock
from theory.runtime.store import write_design

from ..types import FrozenMainPlan, MainProfile
from .design import design


def plan(directory: Path, profile: MainProfile, horizon: float | None = None) -> FrozenMainPlan:
    candidate = design(profile, horizon)
    path = directory / "plan.json"
    frozen: FrozenMainPlan = {**candidate, "plan_fingerprint": fingerprint(candidate)}
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("evidence plan changed; use a new output directory")
    else:
        atomic_json(path, frozen)
    for kind in ("measure", "micro"):
        write_design(directory / kind / "design.json", [t for t in frozen["tasks"] if t["kind"] == kind])
    print(dict(Counter(t["kind"] for t in frozen["tasks"])), flush=True)
    return frozen


def perform(a, root, build_dir, meso, micro):
    """Dispatch evidence stages; source builds and scientific execution stay separate."""
    directory = root / "evidence" / a.profile
    if a.action == "build":
        from smp_meso_bindings import build_binary

        from ..runtime.build import build

        stage = getattr(a, "build_stage", None)
        manifest = build(build_dir, meso, micro, a.backend, link_stage=stage)
        # A build interrupted after the adapter can restore it into a new APFS
        # stage while its immutable manifest still names the original stage.
        a.engine = Path(manifest["binary"])
        binary = a.binary or root / "bin" / "smp-kinetic"
        target = stage / "smp-kinetic" if stage else binary
        build_binary(command_name="kinetic", source_root=meso, output_path=target, backend=a.backend)
        if target != binary:
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, binary)
            if sha(target) != sha(binary):
                raise RuntimeError("staged kinetic binary copy differs")
        # Bind the public binary to actual production source hashes too.
        from common.runtime.go import source_manifest

        atomic_json(
            root / "bin" / "kinetic_manifest.json",
            {
                "binary": str(target),
                "cached_binary": str(binary),
                "binary_sha256": sha(binary),
                "source": source_manifest(meso),
                "backend": a.backend,
                "adapter_source": manifest["engine"],
            },
        )
        if stage:
            # macOS can stall executable mappings on exFAT even when copy hashes
            # match. Execute the small APFS binary; retain the Cache provenance copy.
            a.binary = target
        return
    if a.action in ("plan", "run"):
        candidate = plan(directory, a.profile, a.horizon)
        if a.action == "plan":
            return
        from ..micro_validation.population.run import execute as population
        from ..runtime.measure_runs import execute as measure

        binary = a.binary or root / "bin" / "smp-kinetic"
        manifest = json.loads((build_dir / "manifest.json").read_text())
        engine = Path(getattr(a, "engine", None) or manifest["binary"])
        if sha(engine) != manifest["binary_sha256"]:
            raise ValueError("engine binary differs from build manifest")
        lock = directory / "active.lock"
        store = getattr(a, "store", None)
        store.read(lambda: acquire_lock(lock)) if store else acquire_lock(lock)
        failures = []
        interrupted = False
        try:
            for kind, target, executable in (
                ("measure", directory / "measure", binary),
                ("micro", directory / "micro" / "raw", engine),
            ):
                try:
                    if kind == "measure":
                        measure(
                            [t for t in candidate["tasks"] if t["kind"] == "measure"],
                            target,
                            executable,
                            force=a.force,
                            limit=a.limit,
                            workers=a.workers,
                            store=store,
                        )
                    else:
                        population(
                            [t for t in candidate["tasks"] if t["kind"] == "micro"],
                            target,
                            executable,
                            force=a.force,
                            limit=a.limit,
                            workers=a.workers,
                            store=store,
                        )
                except BatchTaskError as error:
                    failures.extend(error.failures)
        except BaseException as error:
            interrupted = not isinstance(error, Exception)
            raise
        finally:
            if interrupted:
                try:
                    release_lock(lock)
                except OSError:
                    pass  # Resume recovers a stale PID lock after the process exits.
            else:
                store.read(lambda: release_lock(lock)) if store else release_lock(lock)
        if failures:
            raise BatchTaskError(failures)
        return
    if a.action == "analyze":
        from .analyze import analyze

        return analyze(directory)
    if a.action == "plot":
        from .main_figures import plot

        return plot(directory)
    raise ValueError("action unavailable for evidence")
