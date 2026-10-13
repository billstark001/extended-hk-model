"""Current main figures: generation, frozen-data analysis and CSV-only plotting."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "resume", "analyze", "plot", "status", "migrate"))
    p.add_argument("--profile", choices=("main", "main_smoke"))
    p.add_argument("--run-dir", type=Path, help="frozen dataset or bulk run directory")
    p.add_argument("--work-dir", type=Path, help="new bulk run directory")
    p.add_argument("--publish-dir", type=Path)
    p.add_argument("--output-dir", type=Path, help="derived figures or migrated data destination")
    p.add_argument("--control-dir", type=Path, help="persistent local APFS control/spool directory")
    p.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--micro-root", type=Path)
    p.add_argument("--meso-root", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--backend", choices=("purego", "accelerate", "openblas"))
    p.add_argument("--horizon", type=float)
    p.add_argument("--spool-mib", type=int, default=4096, help="aggregate binary/staged-record allowance")
    p.add_argument("--tables-only", action="store_true", help="plot CSV exports without loading raw trajectories")
    return p


def dataset_directory(root):
    root = Path(root).resolve()
    if (root / "plan.json").is_file():
        return root
    candidates = [p.parent for p in (root / "evidence").glob("main*/plan.json")]
    if len(candidates) != 1:
        raise ValueError("expected one frozen main dataset; specify its directory directly")
    return candidates[0]


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.action in ("run", "resume"):
        from .evidence.run_main import launch

        a.resume = a.action == "resume"
        a.work_dir = a.run_dir if a.resume else a.work_dir
        if a.work_dir is None or a.publish_dir is None:
            p.error("run requires --work-dir; resume requires --run-dir; both require --publish-dir")
        return launch(a)
    if a.action == "plan":
        if a.work_dir is None:
            p.error("plan requires --work-dir")
        from .evidence.workflow import plan

        profile = a.profile or "main"
        return plan(a.work_dir / "evidence" / profile, profile, a.horizon)
    if a.run_dir is None:
        p.error(f"{a.action} requires --run-dir")
    directory = dataset_directory(a.run_dir)
    if a.action == "plot":
        if a.tables_only or not (directory / "micro" / "raw").is_dir():
            from .evidence.table_figures import plot_tables

            return plot_tables(directory, a.output_dir)
        from .evidence.main_figures import plot

        return plot(directory)
    if a.action == "analyze":
        from .evidence.analyze import analyze

        return analyze(directory)
    if a.action == "status":
        from common.io.records import record_path
        from theory.runtime.completed import cached

        plan = json.loads((directory / "plan.json").read_text())
        available = 0
        for task in plan["tasks"]:
            raw = directory / task["kind"] / "raw"
            path = raw / (task["id"] + ".npz") if task["kind"] == "measure" else record_path(raw, task["id"])
            if path.exists() and cached(path, task, task["kind"]):
                available += 1
        result = {
            "planned": len(plan["tasks"]),
            "valid_raw_records": available,
            "raw_available": (directory / "micro" / "raw").is_dir(),
            "export_report": json.loads((directory / "report.json").read_text())
            if (directory / "report.json").exists()
            else None,
        }
        print(json.dumps(result, indent=2))
        return result
    if a.output_dir is None:
        p.error("migrate requires a separate --output-dir")
    from .evidence.migrate import migrate

    return migrate(directory, a.output_dir)


if __name__ == "__main__":
    main()
