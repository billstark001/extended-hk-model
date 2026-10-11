"""Build, run and analyze scientific studies through one command interface."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def parser():
    """Build the shared CLI parser; workflow-specific options are forwarded to the selected module."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "build",
            "plan",
            "run",
            "smoke",
            "status",
            "estimate",
            "analyze",
            "plot",
            "pipeline",
            "review",
            "scan",
            "clocks",
            "figures",
            "audit",
        ),
    )
    p.add_argument(
        "--suite",
        choices=(
            "evidence",
            "validation",
            "ensembles",
            "scaling",
            "release",
            "population_closure",
            "population",
            "existing_release",
            "existing_scaling",
        ),
        default="validation",
    )
    p.add_argument("--profile", choices=("minimum", "core", "extended", "paper", "smoke"))
    p.add_argument("--label")
    p.add_argument("--baseline-label")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--build-dir", type=Path)
    p.add_argument("--micro-root", type=Path)
    p.add_argument("--meso-root", type=Path)
    p.add_argument("--backend", choices=("purego", "accelerate", "openblas"), default="purego")
    p.add_argument("--binary", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hours", type=float)
    p.add_argument("--horizon", type=float, help="evidence scaled-time horizon; formal >=120, smoke <=1")
    p.add_argument("--no-wall-budget", action="store_true")
    p.add_argument("--job-timeout", type=float)
    p.add_argument("--families", nargs="+", choices=("micro", "clocks", "operator", "kernel"))
    p.add_argument("--tiers", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--scan-input", type=Path)
    p.add_argument("--clock-input", type=Path)
    p.add_argument(
        "--workflow",
        help="scan: expanded/baseline; clocks: fit/parameter_clocks/plot_clocks/plot_parameter_summary/reduced_mixture/branch_audit/validate/refine",
    )
    return p


def module_call(module, argv):
    """Isolate argparse state for a substantive workflow module."""
    old = sys.argv
    try:
        sys.argv = [module] + argv
        return importlib.import_module(module).main()
    finally:
        sys.argv = old


def perform(a, extra):
    """Dispatch one parsed action; output roots are generated locations while input roots are read-only datasets."""
    from .paths import MESO, MICRO, output_root
    from .runtime.build import build
    from .runtime.paths import OutputPaths

    outputs = OutputPaths(a.output_root)
    build_dir = a.build_dir.expanduser().resolve() if a.build_dir else outputs.build
    micro = a.micro_root.expanduser().resolve() if a.micro_root else MICRO.resolve()
    meso = a.meso_root.expanduser().resolve() if a.meso_root else MESO.resolve()
    root = output_root(a.output_root)
    if a.suite == "evidence":
        from .evidence.workflow import perform as evidence

        if extra:
            raise ValueError("unexpected evidence arguments: " + " ".join(extra))
        return evidence(a, root, build_dir, meso, micro)
    if a.action == "build":
        if a.suite in ("validation", "ensembles", "population"):
            manifest = build(build_dir, meso, micro, a.backend)
            print(json.dumps({k: manifest[k] for k in ("binary", "binary_sha256", "source_fingerprint")}, indent=2))
        else:
            from smp_meso_bindings import build_binary

            destination = a.binary or root / "bin" / "smp-kinetic"
            print(build_binary(command_name="kinetic", source_root=meso, output_path=destination, backend=a.backend))
        return
    if a.action in ("scan", "clocks"):
        for name, value in (
            ("EHK_RELEASE_SCAN_INPUT_DIR", a.scan_input),
            ("EHK_RELEASE_CLOCK_INPUT_DIR", a.clock_input),
            ("EHK_KINETIC_BINARY", a.binary),
        ):
            if value is not None:
                os.environ[name] = str(value.expanduser().resolve())
        if a.action == "scan":
            workflow = a.workflow or "expanded"
            if workflow not in ("expanded", "baseline", "scaling"):
                raise ValueError("unknown scan workflow")
            args = (["--workers", str(a.workers)] if workflow == "expanded" else []) + extra
            if a.output_root:
                args += ["--output", str(root)]
            if a.scan_input:
                args += ["--input-dir", str(a.scan_input)]
            if a.binary:
                args += ["--binary", str(a.binary)]
            return module_call("experiments.paper_theory.energy_scan." + workflow + ".cli", args)
        workflows = {
            "fit": "analyze.driver",
            "parameter_clocks": "parameter_clocks.cli",
            "plot_clocks": "plot_clocks",
            "plot_parameter_summary": "plot_parameter_summary",
            "reduced_mixture": "reduced_mixture",
            "branch_audit": "branch_audit",
            "validate": "validate",
            "refine": "refine",
        }
        if a.output_root:
            os.environ["EHK_CLOCK_OUTPUT_DIR"] = str(root)
        workflow = a.workflow or "fit"
        if workflow not in workflows:
            raise ValueError("unknown clock workflow")
        return module_call("experiments.paper_theory.release_clocks." + workflows[workflow], extra)
    if a.action == "review":
        destination = root / "manuscript" / "reviews" / a.label
        if a.suite == "ensembles":
            if not a.baseline_label:
                raise ValueError("ensemble review requires --baseline-label")
            from .manuscript.reports.ensembles import review

            return review(outputs.root, a.label, a.baseline_label, destination)
        from .manuscript.reports.validation import review

        return review(outputs.control(a.label), destination)
    if a.action in ("figures", "audit"):
        if a.action == "figures":
            module = "figures.paper"
        else:
            workflow = a.workflow or "claims"
            if workflow not in ("claims", "budgets"):
                raise ValueError("unknown audit workflow")
            module = "audit." + workflow
        return module_call(
            "experiments.paper_theory.manuscript." + module, (["--output", str(root)] if a.output_root else []) + extra
        )
    if extra:
        raise ValueError("unexpected arguments: " + " ".join(extra))
    if a.suite in ("validation", "ensembles"):
        if a.action in ("plan", "run", "smoke", "status"):
            from .runtime.runner import main

            args = [
                a.action,
                "--preset",
                a.suite,
                "--profile",
                a.profile,
                "--label",
                a.label,
                "--output-root",
                str(outputs.root),
                "--build-dir",
                str(build_dir),
                "--workers",
                str(a.workers),
                "--hours",
                str(a.hours),
                "--job-timeout",
                str(a.job_timeout),
            ]
            if a.no_wall_budget:
                args += ["--no-wall-budget"]
            if a.families:
                args += ["--families"] + a.families
            if a.tiers:
                args += ["--tiers"] + a.tiers
            return main(args)
        if a.action == "analyze":
            from .analysis import analyze_run

            return analyze_run(outputs.root, a.label, a.families, a.baseline_label)
        if a.action == "estimate":
            from common.runtime.costs import estimate

            from .designs import generate

            baseline = outputs.control(a.baseline_label or "validation")
            plan_path = outputs.control(a.label) / "plan.json"
            if plan_path.exists():
                from theory.runtime.store import read_plan

                jobs = read_plan(plan_path)["jobs"]
            else:
                jobs = generate(
                    preset=a.suite,
                    profile="core" if a.profile == "smoke" else a.profile,
                    families=a.families,
                    smoke=a.profile == "smoke",
                )
            report = estimate(jobs, baseline, a.workers)
            print(json.dumps(report, indent=2))
            return report
        if a.action == "plot":
            # Study analyzers assemble their figures from validated raw records.
            from .analysis import analyze_run

            return analyze_run(outputs.root, a.label, a.families, a.baseline_label)
        raise ValueError("action unavailable for this suite")
    if a.suite == "population":
        from theory.runtime.store import write_design

        from .micro_validation.population.design import designs

        directory = root / "micro_validation" / "population" / a.profile
        if a.action in ("plan", "run", "smoke"):
            tasks = write_design(directory / "design.json", designs(a.profile))
            print(f"{len(tasks)} matched microscopic trajectories")
            if a.action == "plan":
                return
            if a.action != "plan":
                from .micro_validation.population.run import execute

                binary = a.binary or Path(json.loads((build_dir / "manifest.json").read_text())["binary"])
                return execute(tasks, directory / "raw", binary, force=a.force, limit=a.limit)
        if a.action == "analyze":
            from .micro_validation.population.analyze import analyze

            return analyze(directory)
        if a.action == "plot":
            from .micro_validation.population.plot import plot

            return plot(directory)
        raise ValueError("action unavailable for population")
    if a.suite in ("existing_release", "existing_scaling"):
        if a.action == "analyze" and a.suite == "existing_release":
            from .release_clocks.analyze.means import existing_release

            return existing_release(root, a.scan_input, a.clock_input)
        if a.action == "plot":
            if a.suite == "existing_release":
                from .release_clocks.plot_means import plot_existing

                return plot_existing(root)
            from .scaling.plot import plot_runs

            return plot_runs(root, a.suite, a.profile, a.scan_input)
        raise ValueError("existing data supports analyze/plot only")
    from theory.runtime.store import write_design

    from .sweeps import designs

    directory = root / a.suite / a.profile
    if a.action in ("plan", "run", "smoke"):
        tasks = write_design(directory / "design.json", designs(a.suite, a.profile))
        if a.action != "plan":
            from .runtime.measure_runs import execute

            return execute(tasks, directory, a.binary or root / "bin" / "smp-kinetic", force=a.force, limit=a.limit)
        print(f"{len(tasks)} measure tasks")
        return
    if a.action == "analyze":
        from .runtime.measure_analysis import new_runs

        return new_runs(root, a.suite, a.profile)
    if a.action == "plot":
        from .scaling.plot import plot_runs

        return plot_runs(root, a.suite, a.profile)
    raise ValueError("action unavailable for this suite")


def main(argv=None):
    """Parse CLI arguments, configure batch resources and execute the requested action.

    A pipeline builds, freezes a plan, optionally estimates cost, runs and analyzes.
    Ordinary run failures permit partial analysis and retain their exit code;
    interrupts stop subsequent stages and return 130. argv defaults to sys.argv[1:].
    """
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in ("scan", "clocks") and "--workflow" in raw_args and "--help" in raw_args:
        # Show the selected workflow's arguments without creating caches.
        args, extra = parser().parse_known_args([v for v in raw_args if v != "--help"])
        return perform(args, extra + ["--help"])
    p = parser()
    a, extra = p.parse_known_args(argv)
    validation = a.suite in ("validation", "ensembles")
    a.profile = a.profile or ("core" if validation else "paper")
    a.label = a.label or (("smoke_" if a.action == "smoke" or a.profile == "smoke" else "") + a.suite)
    a.hours = 9 if a.hours is None else a.hours
    a.job_timeout = (0 if a.suite == "ensembles" else 1800) if a.job_timeout is None else a.job_timeout
    if a.suite == "ensembles" and "--hours" not in (argv if argv is not None else sys.argv[1:]):
        a.no_wall_budget = True
    if a.limit is not None and a.limit < 1:
        p.error("limit must be positive")
    if not 1 <= a.workers <= 4:
        p.error("workers must be 1..4")
    if a.action == "smoke" and not validation:
        a.profile = "smoke"
    from common.runtime.environment import runtime_environment

    from .paths import output_root

    environment_root = a.output_root or output_root() / "validation"
    environment = runtime_environment(environment_root)
    os.environ.update(environment)
    sys.pycache_prefix = environment["PYTHONPYCACHEPREFIX"]
    if a.action != "pipeline":
        return perform(a, extra)
    # Keep interruption and failure semantics of the old foreground scripts.
    status = 0
    stages = ["build", "plan", "estimate", "run", "analyze"]
    if a.suite in ("evidence", "population", "scaling", "release", "population_closure"):
        stages.append("plot")
    for action in stages:
        if action == "estimate" and not a.baseline_label:
            continue
        a.action = action
        try:
            perform(a, extra)
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
            if code == 130 or action != "run":
                raise
            status = code
        except Exception as error:
            if action == "analyze" and status:
                print(f"analysis failed after run exit {status}: {error}", file=sys.stderr)
                break
            raise
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
