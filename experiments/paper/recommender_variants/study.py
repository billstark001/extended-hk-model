"""Independent scheduler for full-trajectory recommendation implementations."""

import argparse
import json
from pathlib import Path

from common.io.json import sha

from ..recommendation_shared import runtime
from . import design as cfg


def make_plan():
    # Keep the historical on-disk study/profile names and schema unchanged.
    return {
        "schema": cfg.DESIGN_VERSION, "study": "paper_recommendation", "profile": cfg.STUDY,
        "protocol": cfg.design_protocol(), "tasks": cfg.generate_tasks(),
    }


def run_task(binary, directory, task, *, protocol=None, existing=None, on_progress=None, arm_binary_sha256=None):
    configurations = cfg.variant_arms(protocol)
    arms = {
        name: None if spec["recommender"] == cfg.REFERENCE else {"recommender": spec["recommender"], "at": 0}
        for name, spec in configurations.items()
    }
    return {"arms": runtime.paired(
        binary, directory, task, "full", 0, cfg.REFERENCE_HORIZON, arms,
        parameters={name: spec["parameters"] for name, spec in configurations.items()},
        existing=(existing or {}).get("arms"),
        on_progress=(lambda result: on_progress({"arms": result})) if on_progress else None,
        binary_sha256=sha(binary), arm_binary_sha256=arm_binary_sha256,
        observation_steps=cfg.observation_steps,
    )}


def _binding(plan, path, binary_sha):
    binding = {"binary_sha256": binary_sha, "stage_evidence": None}
    saved = json.loads(path.read_text()) if path.is_file() else None
    names = set(cfg.variant_arms(plan["protocol"]))
    if saved is not None:
        hashes = saved.get("arm_binary_sha256")
        if (
            hashes is None or set(hashes) != names
            or any(not isinstance(value, str) or not value for value in hashes.values())
        ):
            raise ValueError("variant binary provenance needs an explicit migrate_plan --smp-binary migration")
        binding["arm_binary_sha256"] = hashes
    else:
        binding["arm_binary_sha256"] = dict.fromkeys(names, binary_sha)
    return binding


def _cache_valid(result, binding):
    expected = binding["arm_binary_sha256"]
    return set(result["arms"]) == set(expected) and all(
        arm.get("binary_sha256", expected[name]) == expected[name] for name, arm in result["arms"].items()
    )


def run_study(directory, binary, workers):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    return runtime.run_dataset(
        directory, binary, workers, plan_factory=make_plan, binding_factory=_binding,
        task_runner=lambda task, plan, binding, previous, progress: run_task(
            binary, directory, task, protocol=plan["protocol"], existing=previous,
            on_progress=progress, arm_binary_sha256=binding["arm_binary_sha256"],
        ),
        result_arms=lambda result: result["arms"].values(), resume=True, cache_valid=_cache_valid,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="Existing or new variants dataset directory.")
    parser.add_argument("--smp-binary", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(make_plan(), indent=2))
        return
    if args.directory is None or args.smp_binary is None:
        parser.error("directory and --smp-binary are required unless --plan-only is selected")
    return run_study(args.directory, args.smp_binary, args.concurrency)


if __name__ == "__main__":
    main()
