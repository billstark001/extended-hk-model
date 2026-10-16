"""Original-paper designs using the shared experiment lifecycle."""

from pathlib import Path

from common.experiments.execution import execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_manifest, freeze_plan
from common.io.json import fingerprint, sha
from ehk.micro.simulation import SimulationAdapter

from . import scenarios as cfg
from .paths import STAT_INPUT


def run_simulations(name, workspace, binary, concurrency=4, *, binding_batch=False):
    workspace, binary = Path(workspace).resolve(), Path(binary).resolve()
    tasks = [{"id": metadata["UniqueName"], "metadata": metadata} for metadata in cfg.generate_tasks(name)]
    control = STAT_INPUT.resolve() / "runtime" / fingerprint({"workspace": str(workspace), "study": name})[:20]
    with session_lock(control), interruptible():
        workflow = Workflow(
            {
                "plan": lambda: freeze_plan(
                    control / "plan.json",
                    {"schema": 1, "study": "paper", "profile": name, "workspace": str(workspace), "tasks": tasks},
                )
            },
            control / "attempt.json",
        )
        workflow.dispatch("plan")
        freeze_manifest(control / "binding.json", {"binary_sha256": sha(binary)})
        workspace.mkdir(parents=True, exist_ok=True)
        journal = TaskJournal(control)

        def execute():
            state = "complete"
            try:
                execute_tasks(
                    tasks,
                    SimulationAdapter(binary, workspace, binding_batch=binding_batch),
                    workers=max(1, concurrency),
                    journal=journal,
                    raise_failures=False,
                )
            except BaseException as error:
                state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                raise
            finally:
                result = journal.finish(tasks, state)
            return result

        workflow.stages["run"] = execute
        return workflow.dispatch("run")
