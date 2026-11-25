"""Original-paper designs using the shared experiment lifecycle."""

import platform
import subprocess
from pathlib import Path

from common.experiments.execution import execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_manifest, freeze_plan
from common.io.json import sha
from common.runtime.git import git_fingerprint
from ehk.micro.simulation import SimulationAdapter

from . import scenarios as cfg


def make_plan(name, *, seed_namespace=cfg.SEED_NAMESPACE):
    return {
        "schema": 2,
        "study": "paper",
        "profile": name,
        "protocol": {
            "seed_namespace": seed_namespace,
            "rng": "ehk-independent-pcg-v1",
            "axes": {key: list(values) for key, values in cfg.SCENARIO_AXES[name].items()},
            "legacy_unique_names": True,
        },
        "tasks": cfg.execution_tasks(name, seed_namespace=seed_namespace),
    }


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
    return {
        "binary_sha256": sha(binary),
        "binary_build_info": build.stdout.strip(),
        "python_version": platform.python_version(),
        "python_source_sha256": sources,
        "python_git": git_fingerprint(root),
    }


def freeze_study(name, workspace, binary, *, seed_namespace=cfg.SEED_NAMESPACE):
    workspace = Path(workspace).resolve()
    control = workspace / ".paper-runtime" / name
    plan = freeze_plan(control / "plan.json", make_plan(name, seed_namespace=seed_namespace))
    freeze_manifest(control / "binding.json", execution_provenance(binary))
    return plan


def run_simulations(name, workspace, binary, concurrency=4, *, binding_batch=False, seed_namespace=cfg.SEED_NAMESPACE):
    workspace, binary = Path(workspace).resolve(), Path(binary).resolve()
    tasks = cfg.execution_tasks(name, seed_namespace=seed_namespace)
    adapter = SimulationAdapter(binary, workspace, binding_batch=binding_batch, strict_metadata=True)
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
                execute_tasks(tasks, adapter, workers=max(1, concurrency), journal=journal, raise_failures=False)
            except BaseException as error:
                state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                raise
            finally:
                result = journal.finish(tasks, state)
            return result

        workflow.stages["run"] = execute
        return workflow.dispatch("run")
