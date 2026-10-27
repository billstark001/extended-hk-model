"""Execution policy and legacy source compatibility, independent of scientific identity."""

from pathlib import Path

from common.runtime.failures import ProtocolError

OPERATIONAL = {
    # Fixed scientific task/model sources remain protected. Optional refinement
    # policy is operational here because its tasks are separately frozen and an
    # existing follow-up plan is always read, never regenerated on resume.
    "experiments/paper_lifted/run.py",
    "experiments/paper_lifted/study.py",
    "experiments/paper_lifted/__main__.py",
    "experiments/paper_lifted/policy.py",
    "experiments/paper_lifted/progress.py",
    "experiments/paper_lifted/bindings.py",
    "experiments/paper_lifted/scheduling.py",
    "experiments/paper_lifted/evidence/costs.py",
    "experiments/paper_lifted/evidence/refinement.py",
    "experiments/paper_lifted/evidence/budget.py",
    "experiments/paper_lifted/evidence/analyze.py",
    "experiments/paper_lifted/evidence/figures.py",
    "src/lifted/runtime/costs.py",
    "src/lifted/runtime/forecast.py",
    "src/common/experiments/progress.py",
    "src/common/experiments/lifecycle.py",
    "src/common/experiments/receipts.py",
    "src/common/experiments/execution.py",
    "src/common/experiments/scheduling.py",
    "src/common/runtime/failures.py",
    "src/common/runtime/budget.py",
    "src/common/runtime/go.py",
    "src/common/runtime/costs.py",
}


def scientific_sources(binding):
    return {name: digest for name, digest in binding.items() if name not in OPERATIONAL}


def validate_sources(saved, current):
    if scientific_sources(saved) != scientific_sources(current):
        raise ProtocolError("scientific source changed; use a new run/plan")
    return {
        name: {"frozen": saved.get(name), "execution": current.get(name)}
        for name in sorted(saved.keys() | current.keys())
        if saved.get(name) != current.get(name)
    }


def control_directory(root, plan):
    from .evidence.protocol import MODEL_ROOT

    root = Path(root).resolve()
    if str(root).startswith("/Volumes/"):
        return MODEL_ROOT / "outputs/runtime" / ("lifted-" + plan["plan_fingerprint"][:20])
    return root.parent / (root.name + "-control")


def reference_workers(plan):
    return plan["budget"].get("reference_workers", plan["budget"].get("workers", 4))
