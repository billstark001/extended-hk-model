"""Manuscript tasks, frozen-data figures and offline energy-slot fits."""

import argparse
import json
from pathlib import Path

from common.experiments.lifecycle import Workflow
from common.experiments.plans import coverage, dataset_directory, read_plan

from .evidence.design import PROFILES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "resume", "analyze", "plot", "status", "fit-slot", "plot-slot"))
    p.add_argument("--profile", choices=PROFILES, help="named research task; main/main_smoke select all five topics")
    p.add_argument(
        "--run-dir", type=Path, help="frozen dataset/bulk run; slot actions accept an energy archive or shard directory"
    )
    p.add_argument("--work-dir", type=Path, help="new bulk run directory")
    p.add_argument("--publish-dir", type=Path)
    p.add_argument("--output-dir", type=Path, help="local analysis bundle or figure destination")
    p.add_argument("--network-dir", type=Path, help="portable network_snapshot analysis export for Figure 1(a)")
    p.add_argument(
        "--intuition-graph",
        choices=("restructured", "event"),
        help="network panel; defaults to restructured when --network-dir is supplied",
    )
    p.add_argument("--control-dir", type=Path, help="persistent local APFS control/spool directory")
    p.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--micro-root", type=Path)
    p.add_argument("--meso-root", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--binding-batch", action="store_true", help="let measure tasks use the mesoscopic binding batch runner"
    )
    p.add_argument("--backend", choices=("purego", "accelerate", "openblas"))
    p.add_argument("--horizon", type=float)
    p.add_argument(
        "--phase",
        choices=("run", "analyze", "plot"),
        action="append",
        help="run/resume stages; repeat to select several (default: all three)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="recompute analysis exports and statistics; scientific raw records are unchanged",
    )
    p.add_argument("--spool-mib", type=int, default=4096, help="aggregate binary/staged-record allowance")
    p.add_argument(
        "--tables-only", action="store_true", help="compatibility flag; all figures now read the portable CSV bundle"
    )
    p.add_argument("--dynamics", choices=("hk", "deffuant"), help="energy-slot update rule")
    p.add_argument("--epsilon", type=float, help="energy-slot confidence radius")
    p.add_argument("--alpha", type=float, help="energy-slot influence parameter")
    p.add_argument("--ratio", type=float, help="energy-slot q/alpha")
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    saved_profile = None
    if a.run_dir and (a.run_dir / "plan.json").is_file():
        saved_profile = json.loads((a.run_dir / "plan.json").read_text()).get("profile")
    if a.profile == "network_snapshot" or saved_profile == "network_snapshot":
        from .evidence.network_snapshot import dispatch

        return dispatch(a)
    if a.action in ("run", "resume"):
        from .run import launch

        a.resume = a.action == "resume"
        a.work_dir = a.run_dir if a.resume else a.work_dir
        if a.work_dir is None or a.publish_dir is None:
            p.error("run requires --work-dir; resume requires --run-dir; both require --publish-dir")
        return launch(a)
    if a.action == "plan":
        if a.work_dir is None:
            p.error("plan requires --work-dir")
        from .study import plan

        profile = a.profile or "main"
        return plan(a.work_dir / "evidence" / profile, profile, a.horizon)
    if a.run_dir is None:
        p.error(f"{a.action} requires --run-dir")
    if a.action in ("fit-slot", "plot-slot"):
        required = ("dynamics", "epsilon", "alpha", "ratio", "output_dir")
        missing = ["--" + name.replace("_", "-") for name in required if getattr(a, name) is None]
        if missing:
            p.error(f"{a.action} requires {' '.join(missing)}")
        from .energy_scan.slot import export_slot

        return export_slot(
            a.run_dir,
            a.output_dir,
            dynamics=a.dynamics,
            alpha=a.alpha,
            epsilon=a.epsilon,
            ratio=a.ratio,
            plot=a.action == "plot-slot",
        )
    directory = dataset_directory(a.run_dir)
    if a.action == "plot":
        from .evidence.figures import plot

        output = a.output_dir or (directory / "figures_csv" if a.tables_only else None)
        if a.network_dir or a.intuition_graph:
            return plot(directory, output, network_dir=a.network_dir, intuition_graph=a.intuition_graph)
        return plot(directory, output)
    if a.action == "analyze":
        from .evidence.analyze import analyze

        return Workflow({"analyze": lambda: analyze(directory, a.output_dir, force=a.force)}).dispatch("analyze")
    if a.action == "status":
        from common.io.records import record_path
        from theory.runtime.completed import cached

        plan = read_plan(directory / "plan.json")
        available = []
        for task in plan["tasks"]:
            raw = directory / task["kind"] / "raw"
            path = raw / (task["id"] + ".npz") if task["kind"] == "measure" else record_path(raw, task["id"])
            if path.exists() and cached(path, task, task["kind"]):
                available.append(task["id"])
        counts = coverage(plan["tasks"], available)
        result = {
            "planned": counts["planned"],
            "valid_raw_records": counts["completed"],
            "raw_available": any((directory / kind / "raw").is_dir() for kind in ("measure", "micro")),
            "analysis_available": (directory / "analysis_manifest.json").is_file(),
            "export_report": json.loads((directory / "report.json").read_text())
            if (directory / "report.json").exists()
            else None,
        }
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()
