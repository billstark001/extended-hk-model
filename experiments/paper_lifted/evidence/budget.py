"""Freeze a new formal design from complete pilot costs without building or running models."""

import json
from pathlib import Path

from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, sha
from lifted.runtime.costs import plan_cost

from .analyze import available, plans, status
from .design import formal_design
from .protocol import sources


def generation_sources(binding):
    """Offline exports, CLI dispatch and cost selection do not change pilot paths."""
    operational = {
        "experiments/paper_lifted/evidence/budget.py",
        "experiments/paper_lifted/run.py",
        "experiments/paper_lifted/__main__.py",
        "experiments/paper_lifted/evidence/analyze.py",
        "experiments/paper_lifted/evidence/figures.py",
        "src/lifted/runtime/costs.py",
    }
    return {name: digest for name, digest in binding.items() if name not in operational}


def freeze(pilot, destination, *, workers=4, hours=7):
    pilot, destination = Path(pilot).resolve(), Path(destination).resolve()
    if not 6 <= hours <= 8 or not 1 <= workers <= 8 or destination == pilot:
        raise ValueError("new formal directory, 6..8 hours and 1..8 workers required")
    plan, tasks = plans(pilot)
    current = sources()
    if (
        plan["profile"] != "pilot"
        or generation_sources(plan["sources"]) != generation_sources(current)
        or status(pilot)["missing"]
    ):
        raise ValueError("a complete pilot under the current source binding is required")
    report = json.loads((pilot / "analysis/report.json").read_text())
    if not report["naive_base_equal_pairs"]:
        raise ValueError("pilot must verify the Random naive/base comparison")
    measurements = []
    for task, result in available(pilot, tasks):
        p = task["parameters"]
        measurements.append(
            dict(
                task=task["id"],
                kind=task["kind"],
                ratio=p["ratio"],
                population=p["population"],
                bins=p["bins"],
                variant=task["variant"],
                layer=task["layer"],
                paths=len(task["replicates"]),
                horizon_steps=task["horizon_steps"],
                source_step=task["source_step"],
                executed_steps=result["executed_steps"],
                elapsed_seconds=result["elapsed_seconds"],
                wall_seconds=result["wall_seconds"],
                unresolved=sum(r["category"] == "unresolved" for r in result["paths"]),
            )
        )
    reserve = 85 * 60  # 45 min independent validation/tails + 40 min offline output/verification.
    best = None
    cache: dict[str, tuple[float, bool]] = {}
    for initials in (4, 8, 12, 16, 24, 32):
        for mothers in (1, 2):
            for micro_paths in (64, 128, 256):
                for meso_paths in (16, 32, 64, 128, 256, 512, 1024, 2048):
                    candidate = formal_design(
                        initials=initials, mothers=mothers, micro_paths=micro_paths, meso_paths=meso_paths, budget={}
                    )
                    estimate = plan_cost(
                        candidate["tasks"], measurements, workers=workers, reserve_seconds=reserve, cache=cache
                    )
                    if 6 <= estimate["expected_hours"] <= hours:
                        # Prefer independent initial coverage and mothers, then inner replication.
                        score = (
                            micro_paths >= 128 and initials >= 8,
                            initials,
                            mothers,
                            micro_paths,
                            min(meso_paths, 1024),
                            -abs(estimate["expected_hours"] - hours),
                        )
                        if best is None or score > best[0]:
                            best = (score, candidate, estimate, initials, mothers, micro_paths, meso_paths)
    if best is None:
        raise ValueError("pilot costs do not support a 6..8 hour candidate; inspect timings before freezing")
    _, chosen, estimate, initials, mothers, micro_paths, meso_paths = best
    pilot_binding = json.loads((pilot / "binding.json").read_text())
    receipt = {
        "target_hours": hours,
        "hard_hours": 8,
        "workers": workers,
        "analysis_reserve_seconds": 40 * 60,
        "followup_reserve_seconds": 45 * 60,
        "pilot_dir": str(pilot),
        "pilot_plan_sha256": sha(pilot / "plan.json"),
        "pilot_binding_sha256": sha(pilot / "binding.json"),
        "backend": pilot_binding["backend"],
        "runtime_binding": {**pilot_binding, "sources": chosen["sources"]},
        "calibration_revisions": {
            name: {"pilot": plan["sources"].get(name), "formal": digest}
            for name, digest in current.items()
            if plan["sources"].get(name) != digest
        },
        "pilot_records": {str(p.relative_to(pilot)): sha(p) for p in (pilot / "raw").glob("*.msgpack.lz4")},
        "initials_per_condition": initials,
        "mothers_per_initial": mothers,
        "micro_paths_per_state": micro_paths,
        "meso_paths_per_state": meso_paths,
        "core_deadline_hours": 8 - (85 * 60) / 3600,
        "estimate": estimate,
        "formal_started": False,
        "deadline_semantics": "stop dispatch before hard deadline; pending work stays missing, never physical censoring",
    }
    chosen["budget"] = receipt
    if destination.exists() and any(
        p.is_file() and p.name not in ("plan.json", "budget.json") for p in destination.rglob("*")
    ):
        raise ValueError("formal destination already contains execution/output files")
    frozen = freeze_plan(destination / "plan.json", chosen)
    atomic_json(destination / "budget.json", receipt)
    # Re-read the authenticated object: no solver/build/executor is called here.
    read_plan(destination / "plan.json")
    return {
        "directory": str(destination),
        "plan_fingerprint": frozen["plan_fingerprint"],
        "tasks": len(frozen["tasks"]),
        **receipt,
    }
