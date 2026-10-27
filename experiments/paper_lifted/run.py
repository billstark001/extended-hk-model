"""Explicit execution boundary; smoke/pilot never construct or dispatch formal work."""

import json
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest, freeze_plan, read_plan
from common.io.json import atomic_json, sha
from common.runtime.budget import wall_budget
from common.runtime.environment import runtime_environment
from common.runtime.failures import BatchTaskError, ProtocolError
from common.runtime.go import build_adapter, engine_manifest, source_manifest
from common.runtime.storage import RunStorage

from .evidence.design import validation_design
from .evidence.protocol import MODEL_ROOT, sources
from .study import LiftedStudy


def calibrated_adapter(root, name, engine, calibration, *, backend, micro, meso):
    """Stage the measured executable, authenticating its sources and module inputs."""
    source = Path(calibration["pilot_dir"]) / "build" / name
    manifest = json.loads((source / "manifest.json").read_text())
    expected = calibration["runtime_binding"]
    if any(manifest[key] != value for key, value in dict(backend=backend, micro=micro, meso=meso).items()):
        raise ProtocolError("calibrated runtime source/backend changed")
    if manifest["engine"] != engine_manifest(engine) or manifest["binary_sha256"] != expected["binaries"][name]:
        raise ProtocolError("calibrated adapter source/binary changed")
    destination = root / "build" / name
    destination.mkdir(parents=True, exist_ok=True)
    for filename, digest in (
        ("paper-" + name + "-engine", manifest["binary_sha256"]),
        ("go.mod", manifest["module_sha256"]),
        ("go.sum", manifest["sum_sha256"]),
    ):
        original, target = source / filename, destination / filename
        if sha(original) != digest or (target.exists() and sha(target) != digest):
            raise ProtocolError("calibrated executable/module hash changed")
        if not target.exists():
            shutil.copy2(original, target)
        if sha(target) != digest:
            raise ProtocolError("calibrated executable/module copy differs")
    binary = str(destination / ("paper-" + name + "-engine"))
    freeze_manifest(destination / "manifest.json", {**manifest, "binary": binary, "cached_binary": binary})
    return binary


def build(root, *, backend, micro_root, meso_root, calibration=None):
    """Reuse theory's initialization instrumentation; build only a thin lifted continuation adapter."""
    selected = {
        "initial": MODEL_ROOT / "experiments/paper_theory/runtime/engine",
        "lifted": MODEL_ROOT / "src/lifted/runtime/engine",
    }
    micro, meso = source_manifest(micro_root), source_manifest(meso_root)
    if calibration:
        pilot = Path(calibration["pilot_dir"])
        if any(
            sha(pilot / name) != calibration[key]
            for name, key in (("plan.json", "pilot_plan_sha256"), ("binding.json", "pilot_binding_sha256"))
        ):
            raise ProtocolError("calibrated pilot plan/binding changed")
    binaries = {}
    for name, engine in selected.items():
        if calibration:
            binaries[name] = calibrated_adapter(
                root, name, engine, calibration, backend=backend, micro=micro, meso=meso
            )
        else:
            manifest = build_adapter(
                root / "build" / name, meso_root, micro_root, backend, engine, binary_name="paper-" + name + "-engine"
            )
            binaries[name] = manifest["binary"]
    binding = {
        "binaries": {k: sha(v) for k, v in binaries.items()},
        "sources": sources(),
        "micro": micro,
        "meso": meso,
        "backend": backend,
    }
    freeze_manifest(root / "binding.json", binding)
    return binaries, binding


def execute(args, root, plan):
    """Execute/recover scientific tasks only; never import analysis or plotting."""
    if plan["sources"] != sources():
        raise ProtocolError("scientific source changed; use a new run/plan")
    if args.action == "run" and plan["profile"] != "formal":
        raise ValueError("run is reserved for an explicitly frozen formal plan; use resume for validation")
    if args.action == "run" and any((root / "raw").glob("*")):
        raise ValueError("formal run already has records; use resume")
    control = root.parent / (root.name + "-control")
    if str(control).startswith("/Volumes/"):
        control = MODEL_ROOT / "outputs/runtime" / ("lifted-" + plan["plan_fingerprint"][:20])
    budget = plan["budget"]
    meter = wall_budget(root / "wall_budget.json", budget["hard_hours"] * 3600) if budget else nullcontext(None)
    with session_lock(control), interruptible(), meter as hard_deadline:
        os.environ.update(runtime_environment(control))
        os.environ["CGO_ENABLED"] = "0" if args.backend == "purego" else "1"
        started = time.monotonic()
        if budget and (args.workers != budget["workers"] or args.backend != budget["backend"]):
            raise ProtocolError("formal workers/backend differ from the calibrated budget")
        deadline = (
            (hard_deadline - budget["analysis_reserve_seconds"] - budget["followup_reserve_seconds"])
            if budget
            else None
        )
        binaries, binding = build(
            root,
            backend=args.backend,
            micro_root=args.micro_root.resolve(),
            meso_root=args.meso_root.resolve(),
            calibration=budget or None,
        )
        if budget and binding != budget["runtime_binding"]:
            raise ProtocolError("formal runtime differs from the measured pilot")
        store = RunStorage(root, control)
        store.recover()
        study = LiftedStudy(root, plan, binaries, binding, store, workers=args.workers, deadline=deadline)
        failure = None
        try:
            result = study.run()
            if not result["missing"] and plan["profile"] == "formal":
                study.deadline = study.adapter.deadline = hard_deadline - budget["analysis_reserve_seconds"]
                result = study.followup()
        except BatchTaskError as error:
            from .evidence.analyze import status

            failure = str(error)
            result = {"missing": status(root)["missing"]}
        receipt = {
            "profile": plan["profile"],
            "elapsed_seconds": time.monotonic() - started,
            "missing": result["missing"],
            "failure": failure,
        }
        atomic_json(root / "attempt.json", receipt)
        return receipt


def launch(args):
    root = args.work_dir.resolve()
    if args.work_dir.is_symlink() or not 1 <= args.workers <= 8:
        raise ValueError("real output directory and 1..8 workers required")
    if str(root).startswith("/Volumes/") and not Path(*root.parts[:3]).is_mount():
        raise ValueError("output volume is not mounted")
    phases = tuple(
        name
        for name in ("run", "analyze", "plot")
        if name in (getattr(args, "phase", None) or ("run", "analyze", "plot"))
    )
    if args.action in ("smoke", "pilot"):
        if "run" not in phases:
            raise ValueError("new smoke/pilot requires the run phase; use analyze/plot or resume for existing data")
        if root.exists() and any(root.iterdir()):
            raise ValueError("smoke/pilot require a fresh directory; use resume for existing work")
        root.mkdir(parents=True, exist_ok=True)
        plan = freeze_plan(root / "plan.json", validation_design(args.action))
    else:
        plan = read_plan(root / "plan.json")
    started = time.monotonic()
    result = execute(args, root, plan) if "run" in phases else {"missing": []}
    if "analyze" in phases:
        from .evidence.analyze import analyze

        report = analyze(root)
        if "run" not in phases:
            result = report
    if "plot" in phases:
        from .evidence.figures import plot

        plot(root / "analysis")
    print(
        json.dumps(
            dict(
                directory=str(root),
                profile=plan["profile"],
                phases=phases,
                missing=len(result["missing"]),
                elapsed_seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )
    return 2 if "run" in phases and result["missing"] else 0
