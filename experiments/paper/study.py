"""Original-paper designs using the shared experiment lifecycle."""

import json
import platform
import subprocess
from pathlib import Path
from ehk.domain import Plan, OriginalProtocol, Binding, GitProvenance, freeze_plan, RANKED_TIE_RULE

from common.experiments.execution import execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import sha, canonical, fingerprint
from common.runtime.git import git_fingerprint
from ehk.micro.simulation import SimulationAdapter

from . import scenarios as cfg
from .analysis.models import AnalysisOutcome


def make_plan(name, *, seed_namespace=cfg.SEED_NAMESPACE) -> Plan[OriginalProtocol]:
    return Plan(schema=2, study="paper", profile=name,
                protocol=OriginalProtocol(seed_namespace=seed_namespace, rng="ehk-independent-pcg-v1",
                                          axes=cfg.SCENARIO_AXES[name], legacy_unique_names=True,
                                          ranked_tie_rule=RANKED_TIE_RULE),
                tasks=tuple(cfg.execution_tasks(name, seed_namespace=seed_namespace)))


def execution_provenance(binary):
    """Bind the executable and actual Python working tree, including new modules."""
    binary = Path(binary).resolve()
    root = Path(__file__).resolve().parents[2]
    sources = {
        str(path.relative_to(root)): sha(path)
        for directory in (root / "src", root / "experiments")
        for path in sorted(directory.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    build = subprocess.run(["go", "version", "-m", str(binary)], text=True, capture_output=True, check=False)
    return Binding(binary_sha256=sha(binary), binary_build_info=build.stdout.strip(),
                   python_version=platform.python_version(), python_source_sha256=sources,
                   python_git=GitProvenance(**git_fingerprint(root)))


def freeze_study(name, workspace, binary, *, seed_namespace=cfg.SEED_NAMESPACE):
    workspace = Path(workspace).resolve()
    control = workspace / ".paper-runtime" / name
    plan = make_plan(name, seed_namespace=seed_namespace)
    plan = freeze_plan(control / "plan.json", plan)
    freeze_manifest(control / "binding.json", execution_provenance(binary))
    return plan


def run_simulations(name, workspace, binary, concurrency=4, *, seed_namespace=cfg.SEED_NAMESPACE):
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("positive simulation concurrency required")
    workspace, binary = Path(workspace).resolve(), Path(binary).resolve()
    tasks = cfg.execution_tasks(name, seed_namespace=seed_namespace)
    adapter = SimulationAdapter(binary, workspace)
    for task in tasks:
        adapter.validate_existing(task)
    control = workspace / ".paper-runtime" / name
    with session_lock(control), interruptible():
        workflow = Workflow(
            {"plan": lambda: freeze_study(name, workspace, binary, seed_namespace=seed_namespace)},
            control / "attempt.json",
        )
        workflow.dispatch("plan")
        workspace.mkdir(parents=True, exist_ok=True)
        journal = TaskJournal(control)

        def execute():
            state = "complete"
            try:
                execute_tasks(tasks, adapter, workers=concurrency, journal=journal, raise_failures=False)
            except BaseException as error:
                state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                raise
            finally:
                result = AnalysisOutcome.from_journal(journal.finish((task.id for task in tasks), state),
                                                     fingerprint(make_plan(name, seed_namespace=seed_namespace)),
                                                     str(workspace))
            return result

        workflow.stages["run"] = execute
        return workflow.dispatch("run")
