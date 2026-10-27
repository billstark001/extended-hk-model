"""Explicit execution boundary; smoke/pilot never construct or dispatch formal work."""

import json
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_plan, read_plan
from common.experiments.receipts import archive_receipt
from common.io.json import atomic_json
from common.runtime.budget import extend_wall_budget, wall_budget
from common.runtime.environment import runtime_environment
from common.runtime.failures import BatchTaskError, ProtocolError
from common.runtime.storage import RunStorage

from . import bindings
from .evidence.design import validation_design
from .evidence.protocol import sources
from .policy import control_directory, reference_workers, validate_sources
from .progress import OverallProgress
from .study import LiftedStudy


def execute(args, root, plan):
    """Execute/recover scientific tasks only; never import analysis or plotting."""
    revisions = validate_sources(plan["sources"], sources())
    if args.action == "run" and plan["profile"] != "formal":
        raise ValueError("run is reserved for an explicitly frozen formal plan; use resume for validation")
    if args.action == "run" and any((root / "raw").glob("*")):
        raise ValueError("formal run already has records; use resume")
    control = control_directory(root, plan)
    budget = plan["budget"]
    if budget and args.backend != budget["backend"]:
        raise ProtocolError("formal backend differs from the calibrated budget")
    with session_lock(control), interruptible():
        seconds = (
            extend_wall_budget(
                root / "wall_budget.json",
                initial_seconds=budget.get("hard_hours", 0 if getattr(args, "wall_hours", None) is not None else 8)
                * 3600,
                seconds=None if getattr(args, "wall_hours", None) is None else args.wall_hours * 3600,
                metadata={"plan_fingerprint": plan["plan_fingerprint"]},
            )
            if (budget or getattr(args, "wall_hours", None) is not None or (root / "wall_budget.json").exists())
            else None
        )
        meter = wall_budget(root / "wall_budget.json", seconds) if seconds is not None else nullcontext(None)
        with meter as hard_deadline:
            return _execute_locked(args, root, plan, control, hard_deadline, revisions)


def _execute_locked(args, root, plan, control, hard_deadline, revisions):
    budget = plan["budget"]
    os.environ.update(runtime_environment(control))
    os.environ["CGO_ENABLED"] = "0" if args.backend == "purego" else "1"
    started = time.monotonic()
    if budget and args.backend != budget["backend"]:
        raise ProtocolError("formal backend differs from the calibrated budget")
    core_only = getattr(args, "core_only", False)
    reserve = budget.get("analysis_reserve_seconds", 0) + (
        0 if core_only else budget.get("followup_reserve_seconds", 0)
    )
    deadline = (hard_deadline - reserve) if hard_deadline is not None else None
    # Preserve the original scientific epoch so old and newly committed records
    # authenticate against the same binding. Actual orchestration changes are audited separately.
    binding_path = root / "binding.json"
    binding_sources = plan["sources"]
    if binding_path.exists() and json.loads(binding_path.read_text())["sources"] != binding_sources:
        raise ProtocolError("original executable binding differs from the frozen source epoch")
    binaries, binding = bindings.build(
        root,
        backend=args.backend,
        micro_root=args.micro_root.resolve(),
        meso_root=args.meso_root.resolve(),
        calibration=budget or None,
        binding_sources=binding_sources,
    )
    if budget and binding != budget["runtime_binding"]:
        raise ProtocolError("formal runtime differs from the measured pilot")
    store = RunStorage(root, control)
    store.recover()
    archive_receipt(root / "attempt.json", root / "executions")
    execution = dict(
        execution_id=uuid.uuid4().hex,
        plan_fingerprint=plan["plan_fingerprint"],
        started_wall=time.time(),
        workers=args.workers,
        reference_workers=reference_workers(plan),
        backend=args.backend,
        schedule=getattr(args, "schedule", "balanced"),
        core_only=core_only,
        core_reserve_seconds=reserve,
        operational_revisions=revisions,
    )
    atomic_json(root / "execution.json", execution)
    atomic_json(root / "executions" / (execution["execution_id"] + ".json"), execution)
    failure = None
    with OverallProgress(
        root, plan, workers=args.workers, core_deadline=deadline, interval=getattr(args, "interval", 15)
    ) as progress:
        study = LiftedStudy(
            root,
            plan,
            binaries,
            binding,
            store,
            workers=args.workers,
            deadline=deadline,
            schedule=execution["schedule"],
            progress=progress,
        )
        try:
            result = study.run()
            if not result["missing"] and plan["profile"] == "formal" and not core_only:
                study.deadline = study.adapter.deadline = hard_deadline - budget["analysis_reserve_seconds"]
                progress.core_deadline = study.deadline
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
        "workers": args.workers,
        "schedule": execution["schedule"],
        "stop_reason": "task_failure" if failure else ("budget" if result["missing"] else "complete"),
    }
    atomic_json(root / "attempt.json", receipt)
    atomic_json(
        root / "executions" / (execution["execution_id"] + ".json"),
        {**execution, **receipt, "finished_wall": time.time()},
    )
    return receipt


def launch(args):
    root = args.work_dir.resolve()
    if args.work_dir.is_symlink():
        raise ValueError("real output directory required")
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
    args.workers = args.workers or reference_workers(plan)
    args.backend = args.backend or plan["budget"].get("backend", "accelerate")
    if not 1 <= args.workers <= 8:
        raise ValueError("1..8 workers required")
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
                stop_reason=result.get("stop_reason"),
                elapsed_seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )
    return 2 if "run" in phases and result["missing"] else 0
