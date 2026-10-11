"""Plan, execute or inspect a study preset with frozen task and build identities."""

import argparse
import collections
import json
import math
import signal
import time
from pathlib import Path

from common.io.json import atomic_json, sha
from common.runtime.process import acquire_lock, release_lock
from theory.runtime.executor import STOP, cancel, run_group
from theory.runtime.store import record_definition_check, resolve_frozen_plan, result_path, summarize

from .. import designs
from ..protocols import ANALYSIS
from .paths import OutputPaths

BUDGET = {"kernel": 0.03, "micro": 0.30, "clocks": 0.40, "operator": 0.27}


def definition_manifest():
    """Hash study definition files for audit; frozen jobs/protocols independently determine numerical compatibility."""
    root = Path(__file__).resolve().parents[1]
    names = ["designs.py", "protocols.py", "runtime/tasks.py"] + [
        f"{p}/design.py" for p in ("micro_validation", "release_clocks", "operator_validation", "kernel_validation")
    ]
    return {name: sha(root / name) for name in names}


def main(argv=None):
    """Plan, inspect or run a preset with frozen plan/build bindings, an exclusive run lock, bounded workers and restored signal handlers."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("validation", "ensembles"), default="validation")
    p.add_argument("action", choices=("plan", "smoke", "run", "status"))
    p.add_argument("--families", nargs="+", choices=designs.FAMILIES, default=None)
    p.add_argument("--tiers", nargs="+", choices=designs.TIERS)
    p.add_argument("--profile", choices=(*designs.PROFILES, "smoke"), default=designs.DEFAULT_PROFILE)
    p.add_argument("--label", help="output label; defaults to the selected study preset")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hours", type=float, default=9)
    p.add_argument(
        "--no-wall-budget", action="store_true", help="finish the frozen design without a global/stage wall deadline"
    )
    p.add_argument("--job-timeout", type=float, default=1800)
    p.add_argument("--output-root", type=Path, help="root for all run metadata, raw results and analysis outputs")
    p.add_argument("--build-dir", type=Path, help="override the build directory under output-root")
    a = p.parse_args(argv)
    default_label = a.preset
    a.label = default_label if a.label is None else a.label
    a.families = (
        list(designs.FAMILIES if a.preset == "validation" else ("micro", "clocks"))
        if a.families is None
        else a.families
    )
    outputs = OutputPaths(a.output_root)
    a.build_dir = outputs.build if a.build_dir is None else a.build_dir.expanduser().resolve()
    if not 1 <= a.workers <= 4:
        p.error("workers must be 1..4 (total processes, no nested parallelism)")
    if not math.isfinite(a.hours) or a.hours <= 0:
        p.error("hours must be positive, or use --no-wall-budget")
    if not a.label.replace("_", "").replace("-", "").isalnum():
        p.error("label must be a simple directory name")
    if not math.isfinite(a.job_timeout) or a.job_timeout < 0:
        p.error("job timeout must be nonnegative; 0 disables it")
    if a.action == "smoke":
        if a.label == default_label:
            a.label = "smoke_" + a.preset
        if not a.label.startswith("smoke_"):
            p.error("smoke label must start with smoke_")
        jobs = designs.generate(
            a.families,
            preset=a.preset,
            profile="core" if a.profile == "smoke" else a.profile,
            tiers=a.tiers,
            smoke=True,
        )
        a.hours = min(a.hours, 1 / 6)
        a.no_wall_budget = False
    else:
        jobs = designs.generate(
            a.families,
            preset=a.preset,
            profile="core" if a.profile == "smoke" else a.profile,
            tiers=a.tiers,
            smoke=a.profile == "smoke",
        )
    print(json.dumps(summarize(jobs), indent=2), flush=True)
    root = outputs.control(a.label)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": 1,
        "profile": a.profile,
        "jobs": jobs,
        "definitions_sha256": definition_manifest(),
        "counts": summarize(jobs),
        "preset": a.preset,
    }
    if a.preset == "ensembles":
        plan["analysis_protocol"] = ANALYSIS
    plan_path = root / "plan.json"
    current_hashes = plan["definitions_sha256"]
    plan, plan_id, definitions_changed = resolve_frozen_plan(
        plan, json.loads(plan_path.read_text()) if plan_path.exists() else None
    )
    jobs = plan["jobs"]
    if definitions_changed:
        print(
            f"Definition file hashes changed; complete generated design is identical. Reusing frozen plan {plan_id}.",
            flush=True,
        )
        if a.action != "status":
            record_definition_check(root, plan_id, plan["definitions_sha256"], current_hashes)
    if a.action == "plan":
        if not plan_path.exists():
            atomic_json(plan_path, {**plan, "plan_fingerprint": plan_id})
        return
    if a.action == "status":
        counts = collections.Counter()
        for j in jobs:
            counts[(j["family"], "complete" if result_path(j, a.label, outputs).exists() else "pending")] += 1
        print(json.dumps({f"{f}/{s}": n for (f, s), n in counts.items()}, indent=2))
        return
    manifest_path = a.build_dir / "manifest.json"
    if not manifest_path.exists():
        p.error("build first with the build command; run never rebuilds or changes source")
    build = json.loads(manifest_path.read_text())
    binary = Path(build["binary"])
    if not binary.exists() or sha(binary) != build["binary_sha256"]:
        raise RuntimeError("binary missing or changed")
    binding = {
        "plan_fingerprint": plan_id,
        "source_fingerprint": build["source_fingerprint"],
        "binary_sha256": build["binary_sha256"],
    }
    bind = root / "binding.json"
    if bind.exists() and json.loads(bind.read_text()) != binding:
        raise RuntimeError("run label bound to a different source or plan")
    if not plan_path.exists():
        atomic_json(plan_path, {**plan, "plan_fingerprint": plan_id})
    atomic_json(bind, binding)
    journal = root / "journal.jsonl"
    lock = root / "active.lock"
    acquire_lock(lock)
    STOP.clear()
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    started = time.monotonic()
    global_deadline = math.inf if a.no_wall_budget else started + a.hours * 3600
    records = []
    try:
        selected = [f for f in BUDGET if any(j["family"] == f for j in jobs)]
        for family in selected:
            group = [j for j in jobs if j["family"] == family]
            # Unused time is carried forward; no stage may exceed global deadline.
            fraction = BUDGET[family] / sum(BUDGET[f] for f in selected[selected.index(family) :])
            deadline = (
                global_deadline
                if family == selected[-1]
                else time.monotonic() + max(0, global_deadline - time.monotonic()) * fraction
            )
            records += run_group(
                group,
                binary,
                build["source_fingerprint"],
                a.label,
                deadline,
                a.workers,
                a.job_timeout,
                journal,
                outputs,
            )
            atomic_json(
                root / "status.json",
                {
                    "binding": binding,
                    "wall_budget_hours": None if a.no_wall_budget else a.hours,
                    "job_timeout_seconds": a.job_timeout or None,
                    "wall_seconds": time.monotonic() - started,
                    "records": records,
                    "counts": dict(collections.Counter(r["status"] for r in records)),
                },
            )
        failed = any(r["status"] in ("failed", "job_timeout") for r in records)
        if STOP.is_set():
            raise SystemExit(130)
        if failed:
            raise SystemExit(1)
    finally:
        release_lock(lock)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
