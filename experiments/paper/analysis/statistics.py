"""SQLite statistics adapter for the common experiment runner."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.orm import undefer

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_plan
from common.io.json import fingerprint
from common.storage.sqlite import create_db_engine_and_session, sync_sqlite_table

from .records import ScenarioStatistics


@dataclass
class StatisticsCompute:
    getter: object
    scenario_path: str
    origin: str

    def __call__(self, job):
        return self.getter(job["metadata"], self.scenario_path, self.origin, job["previous"])


class StatisticsAdapter(TaskAdapter):
    def __init__(self, session, getter, scenario_path, origin, ignore_existing, is_complete=None):
        self.session = session
        self.ignore_existing = ignore_existing
        super().__init__(
            lambda t: t["id"],
            lambda t: (
                t["previous"] is not None
                and (ignore_existing or (is_complete is not None and is_complete(t["previous"])))
            ),
            StatisticsCompute(getter, scenario_path, origin),
            self.persist,
        )

    def previous(self, name, origin):
        row = (
            self.session.query(ScenarioStatistics)
            .filter(ScenarioStatistics.name == name, ScenarioStatistics.origin == origin)
            .options(undefer("*"))
            .first()
        )
        return (
            None
            if row is None
            else SimpleNamespace(**{c.name: getattr(row, c.name) for c in ScenarioStatistics.__table__.columns})
        )

    def persist(self, job, result):
        if result is None:
            return False  # Incomplete source or a study-specific skip; no fabricated result.
        try:
            self.session.merge(result)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise


def analyze_statistics(
    getter,
    scenario_base_path,
    stats_db_path,
    origin,
    scenarios,
    *,
    mode,
    protocol=None,
    is_complete=None,
    ignore_exist=True,
    concurrency=1,
):
    tasks = [{"id": m["UniqueName"], "metadata": m} for m in scenarios]
    database = Path(stats_db_path).resolve()
    control = (
        database.parent / "runtime" / fingerprint({"database": str(database), "origin": origin, "mode": mode})[:20]
    )
    with session_lock(control), interruptible():
        workflow = Workflow(
            {
                "plan": lambda: freeze_plan(
                    control / "plan.json",
                    {
                        "schema": 1,
                        "study": "paper_statistics",
                        "profile": mode,
                        "origin": origin,
                        "workspace": str(Path(scenario_base_path).resolve()),
                        "analysis_protocol": protocol,
                        "tasks": tasks,
                    },
                )
            },
            control / "attempt.json",
        )
        workflow.dispatch("plan")
        engine, session = create_db_engine_and_session(str(database), ScenarioStatistics.Base)
        journal = TaskJournal(control)
        state = "complete"
        try:
            sync_sqlite_table(engine, ScenarioStatistics, extra_columns="error")
            adapter = StatisticsAdapter(session, getter, scenario_base_path, origin, ignore_exist, is_complete)
            jobs = [{**t, "previous": adapter.previous(t["id"], origin)} for t in tasks]

            def execute():
                execute_tasks(
                    jobs,
                    adapter,
                    workers=max(1, concurrency),
                    executor="process",
                    journal=journal,
                    raise_failures=False,
                )
                return journal.finish(tasks)

            workflow.stages["analyze"] = execute
            return workflow.dispatch("analyze")
        except BaseException as error:
            state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            journal.finish(tasks, state)
            session.close()
            engine.dispose()
