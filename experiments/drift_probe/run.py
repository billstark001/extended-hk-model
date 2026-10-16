"""Run temporary frozen-state drift probes on archived microscopic records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow, interruptible, session_lock
from common.experiments.plans import freeze_plan
from common.io.json import sha
from common.runtime.process import run_isolated
from ehk.plotting.landscape import potential_from_force
from experiments.drift_probe.plot import plot_results
from experiments.drift_probe.scenarios import (
    INPUT_DIR,
    NORMALIZED_TIMES,
    OUTPUT_DIR,
    PROBE_BINARY,
    PROBE_RNG,
    TASKS,
    LandscapeScenario,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _probe_steps(active_step: int) -> np.ndarray:
    times = np.asarray(NORMALIZED_TIMES, dtype=float)
    return np.rint(times * active_step).astype(np.int64)


def _result_matrix(state_results: Sequence[Mapping[str, Any]], field: str, component: str | None = None) -> np.ndarray:
    return np.asarray(
        [
            [float(point[component][field] if component else point[field]) for point in state["points"]]
            for state in state_results
        ]
    )


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def run_scenario(
    scenario: LandscapeScenario,
    *,
    workspace: Path,
    probe_binary: Path,
    output_dir: Path,
    grid_step: float,
    replicates: int,
) -> tuple[Path, dict[str, Any]]:
    from smp_bindings import RawSimulationRecord

    metadata = scenario.metadata()
    steps = _probe_steps(scenario.active_step)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = RawSimulationRecord(os.fspath(workspace), metadata)
    with record:
        if scenario.active_step > record.max_step:
            raise ValueError(
                f"{scenario.key}: active step {scenario.active_step} exceeds available step {record.max_step}"
            )
        response = record.evaluate_probe(
            steps=steps.tolist(), h=grid_step, replicates=replicates, rng=PROBE_RNG, binary_path=os.fspath(probe_binary)
        )

    state_results = response["results"]
    returned_steps = np.asarray([int(state["step"]) for state in state_results], dtype=np.int64)
    if not np.array_equal(returned_steps, steps):
        raise RuntimeError(f"{scenario.key}: probe returned steps {returned_steps.tolist()}, expected {steps.tolist()}")

    x = np.asarray([float(point["x"]) for point in state_results[0]["points"]], dtype=float)
    f_probe_mean = _result_matrix(state_results, "mean", "f_probe")
    influence = float(metadata["HKParams"]["Influence"])
    if influence <= 0:
        raise ValueError(
            f"{scenario.key}: HK Influence must be positive to recover the paper's coefficient-free social-force scale"
        )
    f_probe_variance = _result_matrix(state_results, "variance", "f_probe")
    f_neighbor_mean = _result_matrix(state_results, "mean", "f_neighbor")
    f_neighbor_variance = _result_matrix(state_results, "variance", "f_neighbor")
    f_recommendation_mean = _result_matrix(state_results, "mean", "f_recommendation")
    f_recommendation_variance = _result_matrix(state_results, "variance", "f_recommendation")
    social_force_mean = f_probe_mean / influence
    arrays = {
        "scenario_key": np.asarray(scenario.key),
        "scenario_name": np.asarray(metadata["UniqueName"]),
        "label": np.asarray(scenario.label),
        "active_step": np.asarray(scenario.active_step, dtype=np.int64),
        "influence": np.asarray(influence, dtype=float),
        "normalized_time": np.asarray(NORMALIZED_TIMES, dtype=float),
        "actual_normalized_time": returned_steps / scenario.active_step,
        "step": returned_steps,
        "x": x,
        "f_probe_mean": f_probe_mean,
        "f_probe_variance": f_probe_variance,
        "f_probe_samples": _result_matrix(state_results, "samples", "f_probe").astype(np.int64),
        "f_probe_active": _result_matrix(state_results, "active", "f_probe").astype(np.int64),
        "f_neighbor_mean": f_neighbor_mean,
        "f_neighbor_variance": f_neighbor_variance,
        "f_neighbor_samples": _result_matrix(state_results, "samples", "f_neighbor").astype(np.int64),
        "f_neighbor_active": _result_matrix(state_results, "active", "f_neighbor").astype(np.int64),
        "f_recommendation_mean": f_recommendation_mean,
        "f_recommendation_variance": f_recommendation_variance,
        "f_recommendation_samples": _result_matrix(state_results, "samples", "f_recommendation").astype(np.int64),
        "f_recommendation_active": _result_matrix(state_results, "active", "f_recommendation").astype(np.int64),
        # The paper's NOD/social-force convention removes the HK update-rate
        # coefficient. Keep both scales: f_probe is actual one-step drift,
        # while social_force is directly comparable to the existing landscape.
        "social_force_mean": social_force_mean,
        "social_force_variance": f_probe_variance / influence**2,
        "social_force_neighbor_mean": f_neighbor_mean / influence,
        "social_force_neighbor_variance": (f_neighbor_variance / influence**2),
        "social_force_recommendation_mean": (f_recommendation_mean / influence),
        "social_force_recommendation_variance": (f_recommendation_variance / influence**2),
        "mean_concordant_neighbor": _result_matrix(state_results, "mean_concordant_neighbor"),
        "mean_concordant_recommendation": _result_matrix(state_results, "mean_concordant_recommendation"),
        "drift_potential": potential_from_force(x, f_probe_mean),
        "potential": potential_from_force(x, social_force_mean),
        "neighbor_potential": potential_from_force(x, f_neighbor_mean / influence),
    }

    output_path = output_dir / f"counterfactual_probe_{scenario.key}.npz"
    np.savez_compressed(output_path, **arrays)
    manifest_entry = {
        "key": scenario.key,
        "label": scenario.label,
        "unique_name": metadata["UniqueName"],
        "active_step": scenario.active_step,
        "influence": influence,
        "max_step": record.max_step,
        "normalized_time": list(NORMALIZED_TIMES),
        "actual_normalized_time": (returned_steps / scenario.active_step).tolist(),
        "steps": returned_steps.tolist(),
        "result": output_path.name,
        "rng": response["rng"],
    }
    return output_path, manifest_entry


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=TASKS.names,
        dest="scenarios",
        help="task key or all; repeat to select several (default: all)",
    )
    parser.add_argument("--workspace", type=Path, default=INPUT_DIR.resolve())
    parser.add_argument("--probe-binary", type=Path, default=PROBE_BINARY.resolve())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.resolve())
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument(
        "--no-plot", action="store_true", help="save probe arrays without rendering the comparison figure"
    )
    return parser.parse_args(argv)


def probe_task(task):
    return run_scenario(
        LandscapeScenario(**task["scenario"]),
        workspace=Path(task["workspace"]),
        probe_binary=Path(task["binary"]),
        output_dir=Path(task["output"]),
        grid_step=task["grid_step"],
        replicates=task["replicates"],
    )


def run(args) -> None:
    selected = {scenario.key: scenario for name in args.scenarios or ("all",) for scenario in TASKS.select(name)}
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive")
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")
    if not args.probe_binary.is_file():
        raise FileNotFoundError(
            f"probe binary not found: {args.probe_binary}; run `make build-probe` in social-media-models"
        )

    control = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.output_dir / "runtime"))
    tasks = [
        {
            "id": scenario.key,
            "scenario": asdict(scenario),
            "workspace": str(args.workspace.resolve()),
            "binary": str(args.probe_binary.resolve()),
            "output": str(args.output_dir.resolve()),
            "grid_step": args.grid_step,
            "replicates": args.replicates,
        }
        for scenario in selected.values()
    ]
    plan = {
        "schema": 1,
        "study": "drift_probe",
        "tasks": tasks,
        "normalized_times": list(NORMALIZED_TIMES),
        "rng": PROBE_RNG,
        "binary_sha256": sha(args.probe_binary),
    }
    journal = TaskJournal(control)
    results = {}

    def compute(task):
        return run_isolated(probe_task, task)

    def save(task, result):
        results[task["id"]] = result

    def execute():
        state = "complete"
        try:
            execute_tasks(tasks, TaskAdapter(lambda t: t["id"], lambda t: False, compute, save), journal=journal)
        except BaseException as error:
            state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            report = journal.finish(tasks, state)
        return report

    workflow = Workflow(
        {"plan": lambda: freeze_plan(control / "plan.json", plan), "run": execute}, control / "attempt.json"
    )
    workflow.sequence(("plan", "run"))
    result_paths = [results[t["id"]][0] for t in tasks]
    entries = [results[t["id"]][1] for t in tasks]

    social_media_repository = args.probe_binary.resolve().parent
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "frozen-state counterfactual recommendation probe",
        "normalization": (
            "t_n = step / active_step; active_step uses the existing "
            "active_threshold=0.98, min_inactive_value=0.75 definition"
        ),
        "grid": {"minimum": -1.0, "maximum": 1.0, "step": args.grid_step},
        "replicates": args.replicates,
        "common_probe_rng": PROBE_RNG,
        "source_commits": {
            "extended_hk_model": _git_head(REPOSITORY_ROOT),
            "social_media_models": _git_head(social_media_repository),
        },
        "scenarios": entries,
    }
    manifest_path = args.output_dir.resolve() / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"saved {manifest_path}", flush=True)

    if not args.no_plot:
        workflow.stages["plot"] = lambda: plot_results(
            result_paths, args.output_dir.resolve() / "figures" / "counterfactual_probe_landscape"
        )
        figure_paths = workflow.dispatch("plot")
        for path in figure_paths:
            print(f"saved {path}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    with session_lock(args.output_dir / "runtime"), interruptible():
        run(args)


if __name__ == "__main__":
    main()
