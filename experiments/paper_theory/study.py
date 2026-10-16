"""Theory-specific stages registered with the shared experiment lifecycle."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import cast

from common.experiments.execution import execute_tasks
from common.experiments.lifecycle import Workflow, task_lock
from common.experiments.plans import freeze_plan, read_plan, write_design
from common.io.json import atomic_json, sha
from common.runtime.failures import BatchTaskError
from theory.runtime.task_adapters import MeasureAdapter, PopulationAdapter

from .evidence.design import design
from .types import FrozenMainPlan, MainProfile


def plan(directory: Path, profile: MainProfile, horizon: float | None = None, *, resume=False) -> FrozenMainPlan:
    if resume:
        frozen = cast(FrozenMainPlan, read_plan(directory / "plan.json"))
        if frozen["profile"] != profile:
            raise ValueError("frozen plan/profile differs from owning run")
    else:
        frozen = cast(FrozenMainPlan, freeze_plan(directory / "plan.json", design(profile, horizon)))
    for kind in ("measure", "micro"):
        selected = [t for t in frozen["tasks"] if t["kind"] == kind]
        if selected:
            write_design(directory / kind / "design.json", selected)
    print(dict(Counter(t["kind"] for t in frozen["tasks"])), flush=True)
    return frozen


class TheoryStudy:
    def __init__(self, args, root, build_dir, meso, micro):
        self.args, self.root, self.build_dir = args, Path(root), Path(build_dir)
        self.meso, self.micro = meso, micro
        self.directory = self.root / "evidence" / args.profile

    def workflow(self, status_path=None):
        return Workflow(
            {"run": self.run, "analyze": self.analyze, "plot": self.plot},
            status_path,
            resume=getattr(self.args, "resume", False),
            workers=self.args.workers,
        )

    def plan(self):
        return plan(self.directory, self.args.profile, self.args.horizon, resume=getattr(self.args, "resume", False))

    def build(self):
        from smp_meso_bindings import build_binary

        from common.runtime.go import source_manifest

        from .runtime.build import build

        a = self.args
        stage = getattr(a, "build_stage", None)
        manifest = build(self.build_dir, self.meso, self.micro, a.backend, link_stage=stage)
        a.engine = Path(manifest["binary"])
        binary = a.binary or self.root / "bin" / "smp-kinetic"
        target = stage / "smp-kinetic" if stage else binary
        build_binary(command_name="kinetic", source_root=self.meso, output_path=target, backend=a.backend)
        if target != binary:
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, binary)
            if sha(target) != sha(binary):
                raise RuntimeError("staged kinetic binary copy differs")
        atomic_json(
            self.root / "bin" / "kinetic_manifest.json",
            {
                "binary": str(target),
                "cached_binary": str(binary),
                "binary_sha256": sha(binary),
                "source": source_manifest(self.meso),
                "backend": a.backend,
                "adapter_source": manifest["engine"],
            },
        )
        if stage:
            a.binary = target

    def run(self):
        a = self.args
        candidate = self.plan()
        binary = a.binary or self.root / "bin" / "smp-kinetic"
        manifest = json.loads((self.build_dir / "manifest.json").read_text())
        engine = Path(getattr(a, "engine", None) or manifest["binary"])
        if sha(engine) != manifest["binary_sha256"]:
            raise ValueError("engine binary differs from build manifest")
        store = getattr(a, "store", None)
        failures = []
        with task_lock(self.directory, store):
            for kind in ("measure", "micro"):
                selected = [t for t in candidate["tasks"] if t["kind"] == kind]
                if not selected:
                    continue
                adapter = (
                    MeasureAdapter(
                        self.directory / "measure", binary, store, binding_batch=getattr(a, "binding_batch", False)
                    )
                    if kind == "measure"
                    else PopulationAdapter(self.directory / "micro" / "raw", engine, store, force_pairs=a.force)
                )
                try:
                    execute_tasks(selected, adapter, workers=a.workers, force=a.force, limit=a.limit, store=store)
                except BatchTaskError as error:
                    failures.extend(error.failures)
        if failures:
            raise BatchTaskError(failures)

    def analyze(self):
        from .evidence.analyze import analyze

        return analyze(
            self.directory, getattr(self.args, "analysis_dir", None), force=getattr(self.args, "force_analysis", False)
        )

    def plot(self):
        from .evidence.figures import plot

        return plot(getattr(self.args, "analysis_dir", None) or self.directory)
