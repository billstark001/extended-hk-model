"""Independent scheduler for full-trajectory recommendation implementations."""

import argparse
import json
from pathlib import Path

from common.io.json import sha
from ehk.domain import Binding, FullResult, Plan, PolicyChange, parse_binding, to_wire

from ..recommendation_shared import runtime
from . import design as cfg


def make_plan():
    # Keep the historical on-disk study/profile names and schema unchanged.
    return Plan(schema=cfg.DESIGN_VERSION, study="paper_recommendation", profile=cfg.STUDY,
                protocol=cfg.design_protocol(), tasks=tuple(cfg.generate_tasks()))


def run_task(binary, directory, task, *, protocol=None, existing=None, on_progress=None, arm_binary_sha256=None):
    configurations = cfg.variant_arms(protocol)
    arms = {
        name: None if spec.recommender == cfg.REFERENCE else PolicyChange(recommender=spec.recommender, at=0)
        for name, spec in configurations.items()
    }
    return FullResult(arms=runtime.paired(
        binary, directory, task, "full", 0, cfg.REFERENCE_HORIZON, arms,
        parameters={name: spec.parameters for name, spec in configurations.items()},
        existing=existing.arms if existing is not None else None,
        on_progress=(lambda result: on_progress(FullResult(arms=result))) if on_progress else None,
        binary_sha256=sha(binary), arm_binary_sha256=arm_binary_sha256,
        observation_steps=cfg.observation_steps,
    ))


def _binding(plan, path, binary_sha):
    saved = parse_binding(json.loads(path.read_text())) if path.is_file() else None
    names = set(cfg.variant_arms(plan.protocol))
    if saved is not None:
        hashes = saved.arm_binary_sha256
        if (
            hashes is None or set(hashes) != names
            or any(not isinstance(value, str) or not value for value in hashes.values())
        ):
            raise ValueError(
                "variant executable bindings do not match the frozen arm registry; "
                "preserve the original binding and use a new dataset directory"
            )
    else:
        hashes = dict.fromkeys(names, binary_sha)
    return Binding(binary_sha256=binary_sha, stage_evidence=None, arm_binary_sha256=hashes)


def _cache_valid(result, binding):
    expected = binding.arm_binary_sha256
    return expected is not None and set(result.arms) == set(expected) and all(
        arm.binary_sha256 in (None, expected[name]) for name, arm in result.arms.items()
    )


def run_study(directory, binary, workers):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    return runtime.run_dataset(
        directory, binary, workers, plan_factory=make_plan, binding_factory=_binding,
        task_runner=lambda task, plan, binding, previous, progress: run_task(
            binary, directory, task, protocol=plan.protocol, existing=previous,
            on_progress=progress, arm_binary_sha256=binding.arm_binary_sha256,
        ),
        result_arms=lambda result: result.arms.values(), resume=True, cache_valid=_cache_valid,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="Existing or new variants dataset directory.")
    parser.add_argument("--smp-binary", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(to_wire(make_plan()), indent=2))
        return
    if args.directory is None or args.smp_binary is None:
        parser.error("directory and --smp-binary are required unless --plan-only is selected")
    return run_study(args.directory, args.smp_binary, args.concurrency)


if __name__ == "__main__":
    main()
